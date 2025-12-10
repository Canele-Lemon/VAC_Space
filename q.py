    def _load_jacobian_bundle_npy(self):
        """
        bundle["J"]   : (256,3,3)
        bundle["n"]   : (256,)
        bundle["cond"]: (256,)
        """
        if hasattr(self, "_jac_bundle") and self._jac_bundle is not None:
            return
        
        try:
            jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_bundle_ref2744_lam0.001_dw900.0_gs30.0_20251110_105631.npy')
            if not os.path.exists(jac_path):
                raise FileNotFoundError(f"Jacobian npy not found: {jac_path}")

            bundle = np.load(jac_path, allow_pickle=True).item()
            J = np.asarray(bundle["J"], dtype=np.float32) # (256, 3, 3)
            n = np.asarray(bundle["n"], dtype=np.int32)   # (256,)
            cond = np.asarray(bundle["cond"], dtype=np.float32)

            self._jac_bundle = bundle
            self._J_dense = J
            self._J_n = n
            self._J_cond = cond

            logging.info(f"[Jacobian] dense J bundle loaded: {jac_path}, J.shape={J.shape}")

        except Exception:
            logging.exception("[Jacobian] Jacobian load failed")
            raise

이런식으로 Jg를 로드하고 있고 아래처럼 사용하고 있어요

    def _solve_delta_rgb_for_gray(
        self,
        g: int,
        d_targets: dict,
        lam: float = 1e-3,
        # --- (옵션1) 기존처럼 직접 weight 지정하고 싶을 때 ---
        wCx: float | None = None,
        wCy: float | None = None,
        wG:  float | None = None,
        # --- (옵션2) NG 정도에 따라 자동 가중치 계산 ---
        thr_c: float | None = None,
        thr_gamma: float | None = None,
        base_wCx: float = 1.0,
        base_wCy: float = 1.0,
        base_wG:  float = 1.0,
        boost: float = 3.0,
        keep: float = 0.2,
    ):
        """
        주어진 gray g에서, 현재 ΔY = [dCx, dCy, dGamma]를
        자코비안 J_g를 이용해 줄이기 위한 ΔX = [ΔR_H, ΔG_H, ΔB_H]를 푼다.

        관계식:  ΔY_new ≈ ΔY + J_g · ΔX
        우리가 원하는 건 ΔY_new ≈ 0 이므로, J_g · ΔX ≈ -ΔY 를 풀어야 함.

        리지 가중 최소자승:
            argmin_ΔX || W (J_g ΔX + ΔY) ||^2 + λ ||ΔX||^2
            → (J^T W^2 J + λI) ΔX = - J^T W^2 ΔY

        - thr_c, thr_gamma가 주어지면:
            NG 여부에 따라 (base_w * boost) / (base_w * keep)로 가중치 자동 계산
        - thr_c, thr_gamma가 None 이고 wCx/wCy/wG가 주어지면:
            예전 방식처럼 고정 weight 사용
        """
        Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3,3)
        if not np.isfinite(Jg).all():
            logging.warning(f"[BATCH CORR] g={g}: J_g has NaN/inf → skip")
            return None

        dCx_g = float(d_targets["Cx"][g])
        dCy_g = float(d_targets["Cy"][g])
        dG_g  = float(d_targets["Gamma"][g])
        dy = np.array([dCx_g, dCy_g, dG_g], dtype=np.float32)  # (3,)

        # target이 NaN/Inf인 경우
        if not np.isfinite(dy).all():
            logging.warning(
                f"[BATCH CORR] g={g}: dY has NaN/inf "
                f"(dCx, dCy, dG) = ({dCx_g}, {dCy_g}, {dG_g}) → skip this gray"
            )
            return None
        
        # 이미 거의 0이면 굳이 보정 안 해도 됨
        if np.all(np.abs(dy) < 1e-6):
            return None

        # ---------------------------------------------
        # 1) 가중치 계산
        #    - 우선순위:
        #      (1) thr_c/thr_gamma가 있으면 NG 기반 자동 가중치
        #      (2) 아니면 (wCx,wCy,wG) 직접 지정값 사용
        #      (3) 둘 다 없으면 base_w* 그대로 사용
        # ---------------------------------------------
        if thr_c is not None and thr_gamma is not None:
            def w_for(err: float, thr: float, base: float) -> float:
                ratio = abs(err) / max(thr, 1e-6)
                ratio_clamped = min(ratio, 1.0)
                w = base * (keep) + (boost - keep) * ratio_clamped
                return w

            wCx_eff = w_for(dCx_g, thr_c, base_wCx)
            wCy_eff = w_for(dCy_g, thr_c, base_wCy)
            wG_eff  = w_for(dG_g,  thr_gamma, base_wG)

        elif (wCx is not None) and (wCy is not None) and (wG is not None):
            # 옛날 방식: 직접 weight 지정
            wCx_eff, wCy_eff, wG_eff = float(wCx), float(wCy), float(wG)

        else:
            # fallback: 그냥 base weight 사용
            wCx_eff, wCy_eff, wG_eff = base_wCx, base_wCy, base_wG

        w_vec = np.array([wCx_eff, wCy_eff, wG_eff], dtype=np.float32)

        # ---------------------------------------------
        # 2) 가중 least squares (기존 로직 그대로)
        # ---------------------------------------------
        WJ = w_vec[:, None] * Jg   # (3,3)
        Wy = w_vec * dy            # (3,)

        A = WJ.T @ WJ + float(lam) * np.eye(3, dtype=np.float32)  # (3,3)
        b = - WJ.T @ Wy                                           # (3,)

        try:
            dX = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

        step_gain = 1.0
        dR, dG, dB = (float(dX[0]) * step_gain,
                    float(dX[1]) * step_gain,
                    float(dX[2]) * step_gain)

        return dR, dG, dB, wCx_eff, wCy_eff, wG_eff, step_gain
