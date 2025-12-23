def _generate_predicted_vac_lut(
    self,
    base_vac_dict: dict,
    *,
    n_iters: int = 1,
    wG: float = 0.4,          # dGamma weight
    wC: float = 1.0,          # dCx/dCy weight
    lambda_ridge: float = 1e-3,
    use_pattern_onehot: bool = False,   # 지금은 W만 쓸거면 False 권장
    patterns: tuple = ("W",),
    bypass_vac_info_pk: int = 1,        # 질문에서 pk=1이 bypass라고 하셨음
):
    """
    Base VAC(JSON dict)를 입력으로 받아
    1) bypass VAC(pk=1) 대비 ΔLUT(학습 feature와 동일하게) 생성
    2) ML로 dCx/dCy/dGamma per-gray 예측
    3) Jacobian(J_g: 256x3x3)으로 High LUT를 보정 (n_iters 반복)
    4) 256->4096로 업샘플 후 TV write용 JSON 생성

    Returns
    -------
    vac_json_optimized : str | None
    new_lut_4096 : dict | None
    debug_info : dict
    """

    debug_info = {
        "iters": [],
        "bypass_vac_info_pk": bypass_vac_info_pk,
    }

    try:
        # -----------------------------
        # 0) prerequisite check
        # -----------------------------
        if not hasattr(self, "_J_dense") or self._J_dense is None:
            raise RuntimeError("[PredictOpt] Jacobian bundle (_J_dense) not loaded.")

        if not hasattr(self, "models_Y0_bundle") or self.models_Y0_bundle is None:
            raise RuntimeError("[PredictOpt] Prediction models (models_Y0_bundle) not loaded.")

        # -----------------------------
        # 1) mapping index (gray->lut j)
        # -----------------------------
        self._load_mapping_index_gray_to_lut()
        idx_map = np.asarray(self._mapping_index_gray_to_lut, dtype=np.int32)  # (256,)
        if idx_map.shape[0] != 256:
            raise ValueError(f"[PredictOpt] idx_map must be (256,), got {idx_map.shape}")

        # -----------------------------
        # 2) load bypass VAC LUT (4096) from DB (pk=1)
        # -----------------------------
        vac_version_b, bypass_vac_data = self._fetch_vac_by_vac_info_pk(bypass_vac_info_pk)
        if bypass_vac_data is None:
            raise RuntimeError(f"[PredictOpt] bypass VAC fetch failed. pk={bypass_vac_info_pk}")

        bypass_vac_dict = json.loads(bypass_vac_data)

        # -----------------------------
        # 3) extract 4096 LUT arrays (base & bypass)
        # -----------------------------
        def _get_lut4096(d: dict, key: str) -> np.ndarray:
            arr = np.asarray(d[key], dtype=np.float32)
            if arr.shape[0] != 4096:
                raise ValueError(f"[PredictOpt] {key} must be len 4096, got {arr.shape}")
            return arr

        base_RL = _get_lut4096(base_vac_dict, "RchannelLow")
        base_GL = _get_lut4096(base_vac_dict, "GchannelLow")
        base_BL = _get_lut4096(base_vac_dict, "BchannelLow")
        base_RH = _get_lut4096(base_vac_dict, "RchannelHigh")
        base_GH = _get_lut4096(base_vac_dict, "GchannelHigh")
        base_BH = _get_lut4096(base_vac_dict, "BchannelHigh")

        bp_RL = _get_lut4096(bypass_vac_dict, "RchannelLow")
        bp_GL = _get_lut4096(bypass_vac_dict, "GchannelLow")
        bp_BL = _get_lut4096(bypass_vac_dict, "BchannelLow")
        bp_RH = _get_lut4096(bypass_vac_dict, "RchannelHigh")
        bp_GH = _get_lut4096(bypass_vac_dict, "GchannelHigh")
        bp_BH = _get_lut4096(bypass_vac_dict, "BchannelHigh")

        # -----------------------------
        # 4) 256 LUT @ mapped indices
        # -----------------------------
        base_256 = {
            "R_Low":  base_RL[idx_map],
            "G_Low":  base_GL[idx_map],
            "B_Low":  base_BL[idx_map],
            "R_High": base_RH[idx_map],
            "G_High": base_GH[idx_map],
            "B_High": base_BH[idx_map],
        }
        bp_256 = {
            "R_Low":  bp_RL[idx_map],
            "G_Low":  bp_GL[idx_map],
            "B_Low":  bp_BL[idx_map],
            "R_High": bp_RH[idx_map],
            "G_High": bp_GH[idx_map],
            "B_High": bp_BH[idx_map],
        }

        # 초기 제어변수(보정 대상): High만
        high_R = base_256["R_High"].copy()
        high_G = base_256["G_High"].copy()
        high_B = base_256["B_High"].copy()

        # low는 base 그대로 (현재 설계)
        low_R = base_256["R_Low"].copy()
        low_G = base_256["G_Low"].copy()
        low_B = base_256["B_Low"].copy()

        # -----------------------------
        # 5) meta (panel onehot + fr + model_year)
        #    ※ 학습 때 VACInputBuilder meta와 동일한 방식/차원이어야 함
        # -----------------------------
        panel_text, frame_rate, model_year = self._get_ui_meta()

        # 아래 함수는 "학습 때 one-hot 차원/순서"를 그대로 맞춰줘야 합니다.
        # (PANEL_MAKER_CATEGORIES를 쓰든, 앱 내부에 동일한 매핑을 쓰든)
        panel_onehot = self._panel_text_to_onehot(panel_text).astype(np.float32)

        # pattern onehot (옵션)
        pattern_order = list(patterns)
        def _pattern_onehot(p: str) -> np.ndarray:
            v = np.zeros(len(pattern_order), dtype=np.float32)
            if p in pattern_order:
                v[pattern_order.index(p)] = 1.0
            return v

        # -----------------------------
        # 6) helper: build X for model (per-gray)
        #    X schema == VACDataset._build_features_for_gray() 기반
        # -----------------------------
        def _build_X_y0_per_gray(d_lut_256: dict, pat: str = "W") -> np.ndarray:
            """
            d_lut_256:
              keys: R_Low,R_High,G_Low,G_High,B_Low,B_High each (256,)
              값은 "base - bypass" (raw 12bit delta @ mapped indices)
            """
            X_rows = []
            for g in range(256):
                row = [
                    float(d_lut_256["R_Low"][g]),
                    float(d_lut_256["R_High"][g]),
                    float(d_lut_256["G_Low"][g]),
                    float(d_lut_256["G_High"][g]),
                    float(d_lut_256["B_Low"][g]),
                    float(d_lut_256["B_High"][g]),
                ]
                # panel maker onehot
                row.extend(panel_onehot.tolist())
                # fr, model_year
                row.append(float(frame_rate))
                row.append(float(model_year))
                # gray_norm, LUT_j
                row.append(float(g / 255.0))
                row.append(float(idx_map[g]))

                # pattern onehot (선택)
                if use_pattern_onehot:
                    row.extend(_pattern_onehot(pat).tolist())

                X_rows.append(row)

            return np.asarray(X_rows, dtype=np.float32)  # (256, D)

        # -----------------------------
        # 7) helper: ML predict dCx/dCy/dGamma (per-gray)
        # -----------------------------
        def _predict_y0(d_lut_256: dict, pat: str = "W"):
            X = _build_X_y0_per_gray(d_lut_256, pat=pat)

            # payload 구조: {"linear_model":..., "rf_residual":..., "target_scaler":...}
            def _hybrid_predict(model_payload: dict, X: np.ndarray) -> np.ndarray:
                lm = model_payload["linear_model"]
                rf = model_payload["rf_residual"]
                ts = model_payload.get("target_scaler", {"mean": 0.0, "std": 1.0, "standardized": True})
                y_mean = float(ts["mean"])
                y_std  = float(ts["std"])
                standardized = bool(ts.get("standardized", True))

                base_s = lm.predict(X).astype(np.float32)
                resid_s = rf.predict(X).astype(np.float32)
                pred_s = base_s + resid_s

                if standardized:
                    pred = pred_s * y_std + y_mean
                else:
                    pred = pred_s
                return pred.astype(np.float32)

            dCx_pred    = _hybrid_predict(self.models_Y0_bundle["dCx"], X)
            dCy_pred    = _hybrid_predict(self.models_Y0_bundle["dCy"], X)
            dGamma_pred = _hybrid_predict(self.models_Y0_bundle["dGamma"], X)
            return dCx_pred, dCy_pred, dGamma_pred

        # -----------------------------
        # 8) main loop (n_iters)
        # -----------------------------
        for it in range(1, n_iters + 1):

            # 현재 상태에서 feature용 ΔLUT(base - bypass) 구성
            # Low는 base(고정), High는 제어변수(high_R/G/B) 사용
            d_lut_256 = {
                "R_Low":  low_R  - bp_256["R_Low"],
                "G_Low":  low_G  - bp_256["G_Low"],
                "B_Low":  low_B  - bp_256["B_Low"],
                "R_High": high_R - bp_256["R_High"],
                "G_High": high_G - bp_256["G_High"],
                "B_High": high_B - bp_256["B_High"],
            }

            # ML 예측 (per-gray)
            # 지금은 W만 쓰는게 목적이면 patterns=("W",), pat="W"로 고정 추천
            pat = pattern_order[0] if pattern_order else "W"
            dCx_pred, dCy_pred, dGamma_pred = _predict_y0(d_lut_256, pat=pat)

            # Jacobian 보정 (gray별 3x3 solve)
            dh_R = np.zeros(256, dtype=np.float32)
            dh_G = np.zeros(256, dtype=np.float32)
            dh_B = np.zeros(256, dtype=np.float32)

            for g in range(256):
                Jg = self._J_dense[g]  # (3,3)

                if not np.isfinite(Jg).all():
                    continue

                y = np.array([
                    wC * float(dCx_pred[g]),
                    wC * float(dCy_pred[g]),
                    wG * float(dGamma_pred[g]),
                ], dtype=np.float32)

                if not np.isfinite(y).all():
                    continue

                # ridge solve: (J^T J + lam I) dh = - J^T y
                A = (Jg.T @ Jg).astype(np.float32)
                A[np.diag_indices_from(A)] += float(lambda_ridge)
                b = -(Jg.T @ y).astype(np.float32)

                try:
                    dRGB = np.linalg.solve(A, b).astype(np.float32)
                except np.linalg.LinAlgError:
                    continue

                dh_R[g], dh_G[g], dh_B[g] = dRGB[0], dRGB[1], dRGB[2]

            # High LUT 업데이트 (12bit raw)
            high_R = high_R + dh_R
            high_G = high_G + dh_G
            high_B = high_B + dh_B

            # monotone + clip (권장: monotone 먼저)
            high_R = np.clip(self._enforce_monotone(high_R), 0, 4095)
            high_G = np.clip(self._enforce_monotone(high_G), 0, 4095)
            high_B = np.clip(self._enforce_monotone(high_B), 0, 4095)

            debug_info["iters"].append({
                "iter": it,
                "pred_summary": {
                    "dCx_mean": float(np.nanmean(dCx_pred)),
                    "dCy_mean": float(np.nanmean(dCy_pred)),
                    "dGamma_mean": float(np.nanmean(dGamma_pred)),
                    "dCx_abs_mean": float(np.nanmean(np.abs(dCx_pred))),
                    "dCy_abs_mean": float(np.nanmean(np.abs(dCy_pred))),
                    "dGamma_abs_mean": float(np.nanmean(np.abs(dGamma_pred))),
                },
                "dh_summary": {
                    "dR_abs_mean": float(np.nanmean(np.abs(dh_R))),
                    "dG_abs_mean": float(np.nanmean(np.abs(dh_G))),
                    "dB_abs_mean": float(np.nanmean(np.abs(dh_B))),
                }
            })

            logging.info(f"[PredictOpt] iter {it}/{n_iters} done. wG={wG}, wC={wC}, lam={lambda_ridge}")

        # -----------------------------
        # 9) 256 -> 4096 upsample (High only)
        # -----------------------------
        new_lut_4096 = {
            "RchannelLow":  np.clip(np.round(base_RL), 0, 4095).astype(np.uint16),
            "GchannelLow":  np.clip(np.round(base_GL), 0, 4095).astype(np.uint16),
            "BchannelLow":  np.clip(np.round(base_BL), 0, 4095).astype(np.uint16),
            "RchannelHigh": np.clip(np.round(self._up256_to_4096(high_R)), 0, 4095).astype(np.uint16),
            "GchannelHigh": np.clip(np.round(self._up256_to_4096(high_G)), 0, 4095).astype(np.uint16),
            "BchannelHigh": np.clip(np.round(self._up256_to_4096(high_B)), 0, 4095).astype(np.uint16),
        }

        # -----------------------------
        # 10) build json (TV write format)
        # -----------------------------
        vac_json_optimized = self.build_vacparam_std_format(
            base_vac_dict=base_vac_dict,
            new_lut_tvkeys=new_lut_4096
        )

        return vac_json_optimized, new_lut_4096, debug_info

    except Exception:
        logging.exception("[PredictOpt] failed")
        return None, None, debug_info