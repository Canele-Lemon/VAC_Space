    def _run_batch_correction_with_jacobian(self, iter_idx, max_iters, thr_gamma, thr_c, lam=1e-3, metrics=None):

        logging.info(f"[Batch Correction] iteration {iter_idx} start (Jacobian dense)")

        # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
        if not hasattr(self, "_J_dense"):
            logging.error("[Batch Correction] J_dense not loaded") # self._J_dense 없음
            return
        self._load_mapping_index_gray_to_lut()
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.error("[Batch Correction] no VAC cache; need latest TV VAC JSON")
            return

        # 1) NG gray 리스트 / Δ 타깃 준비
        if metrics is not None and "ng_grays" in metrics and "dG" in metrics:
            ng_list = list(metrics["ng_grays"])
            d_targets = {
                "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
                "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
                "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
            }
            thr_gamma = float(metrics.get("thr_gamma", thr_gamma))
            thr_c     = float(metrics.get("thr_c",     thr_c))
            logging.info(f"[Batch Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
        else:
            dG, dCx, dCy, ng_list = SpecEvalThread.compute_gray_errors_and_ng_list(
                self._off_store, self._on_store,
                thr_gamma=thr_gamma, thr_c=thr_c
            )
            d_targets = {
                "Gamma": dG.astype(np.float32),
                "Cx":    dCx.astype(np.float32),
                "Cy":    dCy.astype(np.float32),
            }
            logging.info(f"[Batch Correction] NG grays (recomputed): {ng_list}")

        if not ng_list:
            logging.info("[Batch Correction] no NG gray (또는 0/1/254/255만 NG) → 보정 없음")
            return
    
        # 2) 현재 High LUT 확보
        vac_dict = self._vac_dict_cache

        RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
        GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
        BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

        RH = RH0.copy()
        GH = GH0.copy()
        BH = BH0.copy()

        # 3) index별 Δ 누적 (여러 gray가 같은 index를 참조할 수 있으므로)
        delta_acc = {
            "R": np.zeros_like(RH),
            "G": np.zeros_like(GH),
            "B": np.zeros_like(BH),
        }
        count_acc = {
            "R": np.zeros_like(RH, dtype=np.int32),
            "G": np.zeros_like(GH, dtype=np.int32),
            "B": np.zeros_like(BH, dtype=np.int32),
        }

        mapR = self._lut_map_high["R"]   # (256,)
        mapG = self._lut_map_high["G"]
        mapB = self._lut_map_high["B"]
        
        # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
        for g in ng_list:
            Jg = self._J_dense[g]
            logging.info(f"g={g} | Jg abs min={np.min(np.abs(Jg))} | Jg abs max={np.max(np.abs(Jg))}")
            dX = self._solve_delta_rgb_for_gray(
                g,
                d_targets,
                lam=lam,
                thr_c=thr_c,          # 색좌표 스펙 (예: 0.003)
                thr_gamma=thr_gamma,  # 감마 스펙 (예: 0.05)
                base_wCx=0.5,         # Cx 기본 가중치 (기존 0.5를 base로 사용)
                base_wCy=0.5,         # Cy 기본 가중치
                base_wG=1.0,          # Gamma 기본 가중치
                boost=3.0,            # NG일 때 배율
                keep=0.2,             # OK일 때 배율 (거의 무시)
            )
            if dX is None:
                continue

            dR, dG, dB = dX

            idxR = int(mapR[g])
            idxG = int(mapG[g])
            idxB = int(mapB[g])

            if 0 <= idxR < len(RH):
                delta_acc["R"][idxR] += dR
                count_acc["R"][idxR] += 1
            if 0 <= idxG < len(GH):
                delta_acc["G"][idxG] += dG
                count_acc["G"][idxG] += 1
            if 0 <= idxB < len(BH):
                delta_acc["B"][idxB] += dB
                count_acc["B"][idxB] += 1

        # 5) index별 평균 Δ 적용 + clip + monotone + 로그
        for ch, arr, arr0 in (
            ("R", RH, RH0),
            ("G", GH, GH0),
            ("B", BH, BH0),
        ):
            da = delta_acc[ch]
            ct = count_acc[ch]
            mask = ct > 0

            if not np.any(mask):
                logging.info(f"[Batch Correction] channel {ch}: no indices updated")
                continue

            # 평균 Δ
            arr[mask] = arr0[mask] + (da[mask] / ct[mask])
            # clip
            arr[:] = np.clip(arr, 0.0, 4095.0)
            # 단조 증가 (i<j → LUT[i] ≤ LUT[j])
            self._enforce_monotone(arr)

            # 인덱스별 보정 로그 (before → after)
            changed_idx = np.where(mask)[0]
            logging.info(f"[Batch Correction] channel {ch}: {len(changed_idx)} indices updated")
            for idx in changed_idx:
                before = float(arr0[idx])
                after  = float(arr[idx])
                delta  = after - before
                logging.debug(
                    f"[Batch Correction] ch={ch} idx={idx:4d}: {before:7.1f} → {after:7.1f} (Δ={delta:+.2f})"
                )

        # 6) NG gray 기준으로 어떤 LUT index가 어떻게 바뀌었는지 추가 요약 로그
        for g in ng_list:
            idxR = int(mapR[g])
            idxG = int(mapG[g])
            idxB = int(mapB[g])
            info = []
            if 0 <= idxR < len(RH0):
                info.append(
                    f"R(idx={idxR}): {RH0[idxR]:.1f}→{RH[idxR]:.1f} (Δ={RH[idxR]-RH0[idxR]:+.1f})"
                )
            if 0 <= idxG < len(GH0):
                info.append(
                    f"G(idx={idxG}): {GH0[idxG]:.1f}→{GH[idxG]:.1f} (Δ={GH[idxG]-GH0[idxG]:+.1f})"
                )
            if 0 <= idxB < len(BH0):
                info.append(
                    f"B(idx={idxB}): {BH0[idxB]:.1f}→{BH[idxB]:.1f} (Δ={BH[idxB]-BH0[idxB]:+.1f})"
                )
            if info:
                logging.info(f"[Batch Correction] g={g:3d} → " + " | ".join(info))

        # 7) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "RchannelHigh": RH,
            "GchannelHigh": GH,
            "BchannelHigh": BH,
        }
        for k in new_lut_4096:
            new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

        # UI용 플롯 dict
        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],
            "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],
            "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],
            "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)

        # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
        logging.info(f"[Correction] LUT {iter_idx}차 보정 완료")

        vac_write_json = self.build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )
        vac_dict = json.loads(vac_write_json)
        self._vac_dict_cache = vac_dict

        def _after_write(ok, msg):
            logging.info(f"[VAC Writing] write result: {ok} {msg}")
            if not ok:
                return
            logging.info("[VAC Reading] TV reading after write")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'exit')
            if not vac_dict_after:
                logging.error("[VAC Reading] TV read-back failed")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=vac_dict, read_data=vac_dict_after)
            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")            
            self._step_done(3)
            
            self._fine_mode = False

            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_corr = SessionProfile(
                legend_text=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7,
                            "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr
                self._update_last_on_lv_norm(store_corr)
                
                self._step_start(5)
                self._spec_thread = SpecEvalThread(
                    self._off_store, self._on_store,
                    thr_gamma=thr_gamma, thr_c=thr_c, parent=self
                )
                self._spec_thread.finished.connect(
                    lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx, max_iters)
                )
                self._spec_thread.start()

            logging.info("[BATCH CORR] re-measure start (after LUT update)")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_corr,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                gamma_settle_ms=1000,
                cs_settle_ms=1000,
                on_done=_after_corr
            )

        self._step_start(3)
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)

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
            # NG 기반: 스펙 넘어가면 boost, 아니면 keep
            def w_for(err: float, thr: float, base: float) -> float:
                if abs(err) > thr:
                    return base * boost     # NG → 더 강하게
                else:
                    return base * keep      # OK → 거의 무시 수준

            wCx_eff = w_for(dCx_g, thr_c,     base_wCx)
            wCy_eff = w_for(dCy_g, thr_c,     base_wCy)
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

        step_gain = 16.0
        dR, dG, dB = (float(dX[0]) * step_gain,
                    float(dX[1]) * step_gain,
                    float(dX[2]) * step_gain)

        logging.debug(
            f"[BATCH CORR] g={g}: "
            f"dCx={dCx_g:+.6f}, dCy={dCy_g:+.6f}, dG={dG_g:+.6f} → "
            f"wCx={wCx_eff:.3f}, wCy={wCy_eff:.3f}, wG={wG_eff:.3f} → "
            f"ΔR_H={dR:+.3f}, ΔG_H={dG:+.3f}, ΔB_H={dB:+.3f}"
        )
        return dR, dG, dB
여기서 보정값들
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  2 → R(idx=16): 23.0→nan (Δ=+nan) | G(idx=16): 23.0→nan (Δ=+nan) | B(idx=16): 23.0→nan (Δ=+nan)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  3 → R(idx=36): 51.0→51.0 (Δ=+0.0) | G(idx=36): 51.0→50.8 (Δ=-0.2) | B(idx=36): 51.0→51.7 (Δ=+0.7)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  4 → R(idx=52): 74.0→72.0 (Δ=-2.0) | G(idx=52): 74.0→72.0 (Δ=-2.0) | B(idx=52): 74.0→72.0 (Δ=-2.0)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  5 → R(idx=72): 102.0→102.2 (Δ=+0.2) | G(idx=72): 102.0→102.6 (Δ=+0.6) | B(idx=72): 102.0→101.0 (Δ=-1.0)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  6 → R(idx=92): 131.0→131.1 (Δ=+0.1) | G(idx=92): 131.0→131.3 (Δ=+0.3) | B(idx=92): 131.0→130.0 (Δ=-1.0)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  7 → R(idx=112): 159.0→159.1 (Δ=+0.1) | G(idx=112): 159.0→159.3 (Δ=+0.3) | B(idx=112): 159.0→158.2 (Δ=-0.8)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  8 → R(idx=112): 159.0→159.1 (Δ=+0.1) | G(idx=112): 159.0→159.3 (Δ=+0.3) | B(idx=112): 159.0→158.2 (Δ=-0.8)
2025-11-10 13:12:44,161 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=  9 → R(idx=128): 182.0→182.1 (Δ=+0.1) | G(idx=128): 182.0→182.2 (Δ=+0.2) | B(idx=128): 182.0→181.5 (Δ=-0.5)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 10 → R(idx=148): 210.0→212.9 (Δ=+2.9) | G(idx=148): 210.0→216.1 (Δ=+6.1) | B(idx=148): 210.0→211.2 (Δ=+1.2)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 11 → R(idx=168): 238.0→241.1 (Δ=+3.1) | G(idx=168): 238.0→244.5 (Δ=+6.5) | B(idx=168): 238.0→239.1 (Δ=+1.1)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 12 → R(idx=184): 261.0→265.5 (Δ=+4.5) | G(idx=184): 261.0→270.2 (Δ=+9.2) | B(idx=184): 261.0→263.6 (Δ=+2.6)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 13 → R(idx=204): 289.0→293.4 (Δ=+4.4) | G(idx=204): 289.0→298.3 (Δ=+9.3) | B(idx=204): 289.0→291.3 (Δ=+2.3)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 14 → R(idx=224): 317.0→320.9 (Δ=+3.9) | G(idx=224): 317.0→325.2 (Δ=+8.2) | B(idx=224): 317.0→319.4 (Δ=+2.4)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 15 → R(idx=240): 340.0→344.1 (Δ=+4.1) | G(idx=240): 340.0→348.8 (Δ=+8.8) | B(idx=240): 340.0→342.6 (Δ=+2.6)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 16 → R(idx=240): 340.0→344.1 (Δ=+4.1) | G(idx=240): 340.0→348.8 (Δ=+8.8) | B(idx=240): 340.0→342.6 (Δ=+2.6)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 17 → R(idx=260): 368.0→371.5 (Δ=+3.5) | G(idx=260): 368.0→375.4 (Δ=+7.4) | B(idx=260): 368.0→370.1 (Δ=+2.1)
2025-11-10 13:12:44,162 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 18 → R(idx=279): 395.0→398.8 (Δ=+3.8) | G(idx=279): 395.0→403.0 (Δ=+8.0) | B(idx=279): 395.0→396.5 (Δ=+1.5)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 19 → R(idx=295): 418.0→421.4 (Δ=+3.4) | G(idx=295): 418.0→425.4 (Δ=+7.4) | B(idx=295): 418.0→419.3 (Δ=+1.3)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 20 → R(idx=314): 445.0→447.8 (Δ=+2.8) | G(idx=314): 445.0→451.0 (Δ=+6.0) | B(idx=314): 445.0→446.2 (Δ=+1.2)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 21 → R(idx=334): 474.0→476.5 (Δ=+2.5) | G(idx=334): 474.0→479.3 (Δ=+5.3) | B(idx=334): 474.0→475.7 (Δ=+1.7)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 22 → R(idx=349): 495.0→497.6 (Δ=+2.6) | G(idx=349): 495.0→500.4 (Δ=+5.4) | B(idx=349): 495.0→496.2 (Δ=+1.2)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 23 → R(idx=349): 495.0→497.6 (Δ=+2.6) | G(idx=349): 495.0→500.4 (Δ=+5.4) | B(idx=349): 495.0→496.2 (Δ=+1.2)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 24 → R(idx=368): 522.0→524.5 (Δ=+2.5) | G(idx=368): 522.0→527.3 (Δ=+5.3) | B(idx=368): 522.0→523.0 (Δ=+1.0)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 25 → R(idx=388): 550.0→552.0 (Δ=+2.0) | G(idx=388): 550.0→554.2 (Δ=+4.2) | B(idx=388): 550.0→550.7 (Δ=+0.7)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 26 → R(idx=403): 572.0→573.7 (Δ=+1.7) | G(idx=403): 572.0→575.6 (Δ=+3.6) | B(idx=403): 572.0→572.7 (Δ=+0.7)
2025-11-10 13:12:44,163 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 27 → R(idx=423): 600.0→601.6 (Δ=+1.6) | G(idx=423): 600.0→603.3 (Δ=+3.3) | B(idx=423): 600.0→600.7 (Δ=+0.7)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 28 → R(idx=443): 628.0→629.4 (Δ=+1.4) | G(idx=443): 628.0→630.8 (Δ=+2.8) | B(idx=443): 628.0→628.8 (Δ=+0.8)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 29 → R(idx=459): 651.0→652.3 (Δ=+1.3) | G(idx=459): 651.0→653.8 (Δ=+2.8) | B(idx=459): 651.0→651.6 (Δ=+0.6)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 37 → R(idx=591): 835.0→835.0 (Δ=+0.0) | G(idx=591): 835.0→835.0 (Δ=-0.0) | B(idx=591): 835.0→835.1 (Δ=+0.1)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 38 → R(idx=610): 861.0→861.0 (Δ=+0.0) | G(idx=610): 861.0→861.0 (Δ=-0.0) | B(idx=610): 861.0→861.1 (Δ=+0.1)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 39 → R(idx=610): 861.0→861.0 (Δ=+0.0) | G(idx=610): 861.0→861.0 (Δ=-0.0) | B(idx=610): 861.0→861.1 (Δ=+0.1)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 40 → R(idx=626): 883.0→883.0 (Δ=-0.0) | G(idx=626): 883.0→883.0 (Δ=-0.0) | B(idx=626): 883.0→883.1 (Δ=+0.1)
2025-11-10 13:12:44,164 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 41 → R(idx=646): 910.0→910.0 (Δ=-0.0) | G(idx=646): 910.0→910.0 (Δ=-0.0) | B(idx=646): 910.0→910.1 (Δ=+0.1)
2025-11-10 13:12:44,165 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 42 → R(idx=666): 938.0→938.0 (Δ=-0.0) | G(idx=666): 938.0→938.0 (Δ=-0.0) | B(idx=666): 938.0→938.1 (Δ=+0.1)
2025-11-10 13:12:44,165 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 43 → R(idx=682): 959.0→959.0 (Δ=-0.0) | G(idx=682): 959.0→959.0 (Δ=-0.0) | B(idx=682): 959.0→959.2 (Δ=+0.2)
2025-11-10 13:12:44,165 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 44 → R(idx=702): 987.0→987.0 (Δ=-0.0) | G(idx=702): 987.0→987.0 (Δ=-0.0) | B(idx=702): 987.0→987.2 (Δ=+0.2)
2025-11-10 13:12:44,165 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 45 → R(idx=722): 1014.0→1014.0 (Δ=-0.0) | G(idx=722): 1014.0→1014.0 (Δ=-0.0) | B(idx=722): 1014.0→1014.1 (Δ=+0.1)
2025-11-10 13:12:44,165 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 46 → R(idx=722): 1014.0→1014.0 (Δ=-0.0) | G(idx=722): 1014.0→1014.0 (Δ=-0.0) | B(idx=722): 1014.0→1014.1 (Δ=+0.1)
2025-11-10 13:12:44,165 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 47 → R(idx=738): 1036.0→1036.0 (Δ=-0.0) | G(idx=738): 1036.0→1036.0 (Δ=-0.0) | B(idx=738): 1036.0→1036.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 48 → R(idx=758): 1063.0→1063.0 (Δ=-0.0) | G(idx=758): 1063.0→1063.0 (Δ=-0.0) | B(idx=758): 1063.0→1063.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 49 → R(idx=778): 1090.0→1090.0 (Δ=-0.0) | G(idx=778): 1090.0→1090.0 (Δ=-0.0) | B(idx=778): 1090.0→1090.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 50 → R(idx=794): 1112.0→1112.0 (Δ=-0.0) | G(idx=794): 1112.0→1112.0 (Δ=-0.0) | B(idx=794): 1112.0→1112.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 52 → R(idx=833): 1165.0→1165.0 (Δ=-0.0) | G(idx=833): 1165.0→1165.0 (Δ=-0.0) | B(idx=833): 1165.0→1165.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 53 → R(idx=833): 1165.0→1165.0 (Δ=-0.0) | G(idx=833): 1165.0→1165.0 (Δ=-0.0) | B(idx=833): 1165.0→1165.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 54 → R(idx=849): 1187.0→1187.0 (Δ=-0.0) | G(idx=849): 1187.0→1187.0 (Δ=-0.0) | B(idx=849): 1187.0→1187.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 55 → R(idx=869): 1215.0→1215.0 (Δ=-0.0) | G(idx=869): 1215.0→1215.0 (Δ=-0.0) | B(idx=869): 1215.0→1215.1 (Δ=+0.1)
2025-11-10 13:12:44,166 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 94 → R(idx=1502): 2134.0→2133.6 (Δ=-0.4) | G(idx=1502): 2134.0→2133.2 (Δ=-0.8) | B(idx=1502): 2134.0→2133.4 (Δ=-0.6)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 95 → R(idx=1522): 2163.0→2162.6 (Δ=-0.4) | G(idx=1522): 2163.0→2162.1 (Δ=-0.9) | B(idx=1522): 2163.0→2162.4 (Δ=-0.6)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 96 → R(idx=1522): 2163.0→2162.6 (Δ=-0.4) | G(idx=1522): 2163.0→2162.1 (Δ=-0.9) | B(idx=1522): 2163.0→2162.4 (Δ=-0.6)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 97 → R(idx=1537): 2182.0→2181.6 (Δ=-0.4) | G(idx=1537): 2182.0→2181.1 (Δ=-0.9) | B(idx=1537): 2182.0→2181.4 (Δ=-0.6)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 98 → R(idx=1557): 2207.0→2206.6 (Δ=-0.4) | G(idx=1557): 2207.0→2206.1 (Δ=-0.9) | B(idx=1557): 2207.0→2206.5 (Δ=-0.5)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g= 99 → R(idx=1577): 2231.0→2230.6 (Δ=-0.4) | G(idx=1577): 2231.0→2230.2 (Δ=-0.8) | B(idx=1577): 2231.0→2230.4 (Δ=-0.6)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=100 → R(idx=1597): 2256.0→2255.6 (Δ=-0.4) | G(idx=1597): 2256.0→2255.2 (Δ=-0.8) | B(idx=1597): 2256.0→2255.7 (Δ=-0.3)
2025-11-10 13:12:44,167 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=101 → R(idx=1613): 2276.0→2275.6 (Δ=-0.4) | G(idx=1613): 2276.0→2275.2 (Δ=-0.8) | B(idx=1613): 2276.0→2275.4 (Δ=-0.6)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=165 → R(idx=2643): 3364.0→3364.2 (Δ=+0.2) | G(idx=2643): 3364.0→3364.5 (Δ=+0.5) | B(idx=2643): 3364.0→3364.8 (Δ=+0.8)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=166 → R(idx=2663): 3377.0→3377.2 (Δ=+0.2) | G(idx=2663): 3377.0→3377.5 (Δ=+0.5) | B(idx=2663): 3377.0→3377.8 (Δ=+0.8)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=167 → R(idx=2683): 3391.0→3391.3 (Δ=+0.3) | G(idx=2683): 3391.0→3391.6 (Δ=+0.6) | B(idx=2683): 3391.0→3392.2 (Δ=+1.2)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=168 → R(idx=2699): 3402.0→3402.3 (Δ=+0.3) | G(idx=2699): 3402.0→3402.6 (Δ=+0.6) | B(idx=2699): 3402.0→3403.3 (Δ=+1.3)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=169 → R(idx=2699): 3402.0→3402.3 (Δ=+0.3) | G(idx=2699): 3402.0→3402.6 (Δ=+0.6) | B(idx=2699): 3402.0→3403.3 (Δ=+1.3)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=170 → R(idx=2719): 3416.0→3416.3 (Δ=+0.3) | G(idx=2719): 3416.0→3416.6 (Δ=+0.6) | B(idx=2719): 3416.0→3417.0 (Δ=+1.0)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=171 → R(idx=2739): 3429.0→3429.3 (Δ=+0.3) | G(idx=2739): 3429.0→3429.7 (Δ=+0.7) | B(idx=2739): 3429.0→3430.1 (Δ=+1.1)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=172 → R(idx=2756): 3441.0→3441.3 (Δ=+0.3) | G(idx=2756): 3441.0→3441.7 (Δ=+0.7) | B(idx=2756): 3441.0→3442.5 (Δ=+1.5)
2025-11-10 13:12:44,168 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=173 → R(idx=2776): 3454.0→3454.4 (Δ=+0.4) | G(idx=2776): 3454.0→3454.8 (Δ=+0.8) | B(idx=2776): 3454.0→3455.0 (Δ=+1.0)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=174 → R(idx=2796): 3468.0→3468.4 (Δ=+0.4) | G(idx=2796): 3468.0→3468.8 (Δ=+0.8) | B(idx=2796): 3468.0→3469.8 (Δ=+1.8)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=175 → R(idx=2812): 3479.0→3479.4 (Δ=+0.4) | G(idx=2812): 3479.0→3479.8 (Δ=+0.8) | B(idx=2812): 3479.0→3480.3 (Δ=+1.3)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=176 → R(idx=2812): 3479.0→3479.4 (Δ=+0.4) | G(idx=2812): 3479.0→3479.8 (Δ=+0.8) | B(idx=2812): 3479.0→3480.3 (Δ=+1.3)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=177 → R(idx=2832): 3493.0→3493.4 (Δ=+0.4) | G(idx=2832): 3493.0→3493.9 (Δ=+0.9) | B(idx=2832): 3493.0→3495.4 (Δ=+2.4)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=178 → R(idx=2852): 3506.0→3506.4 (Δ=+0.4) | G(idx=2852): 3506.0→3506.9 (Δ=+0.9) | B(idx=2852): 3506.0→3508.5 (Δ=+2.5)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=179 → R(idx=2868): 3517.0→3517.4 (Δ=+0.4) | G(idx=2868): 3517.0→3517.9 (Δ=+0.9) | B(idx=2868): 3517.0→3519.8 (Δ=+2.8)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=180 → R(idx=2889): 3532.0→3532.4 (Δ=+0.4) | G(idx=2889): 3532.0→3532.9 (Δ=+0.9) | B(idx=2889): 3532.0→3534.3 (Δ=+2.3)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=181 → R(idx=2909): 3545.0→3545.4 (Δ=+0.4) | G(idx=2909): 3545.0→3545.9 (Δ=+0.9) | B(idx=2909): 3545.0→3546.8 (Δ=+1.8)
2025-11-10 13:12:44,169 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=182 → R(idx=2925): 3556.0→3556.4 (Δ=+0.4) | G(idx=2925): 3556.0→3556.9 (Δ=+0.9) | B(idx=2925): 3556.0→3558.1 (Δ=+2.1)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=183 → R(idx=2945): 3570.0→3570.5 (Δ=+0.5) | G(idx=2945): 3570.0→3571.0 (Δ=+1.0) | B(idx=2945): 3570.0→3572.7 (Δ=+2.7)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=184 → R(idx=2945): 3570.0→3570.5 (Δ=+0.5) | G(idx=2945): 3570.0→3571.0 (Δ=+1.0) | B(idx=2945): 3570.0→3572.7 (Δ=+2.7)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=185 → R(idx=2965): 3584.0→3584.5 (Δ=+0.5) | G(idx=2965): 3584.0→3585.0 (Δ=+1.0) | B(idx=2965): 3584.0→3587.7 (Δ=+3.7)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=186 → R(idx=2981): 3595.0→3595.5 (Δ=+0.5) | G(idx=2981): 3595.0→3596.0 (Δ=+1.0) | B(idx=2981): 3595.0→3598.5 (Δ=+3.5)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=187 → R(idx=3001): 3608.0→3608.5 (Δ=+0.5) | G(idx=3001): 3608.0→3609.0 (Δ=+1.0) | B(idx=3001): 3608.0→3611.6 (Δ=+3.6)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=188 → R(idx=3021): 3622.0→3622.5 (Δ=+0.5) | G(idx=3021): 3622.0→3623.0 (Δ=+1.0) | B(idx=3021): 3622.0→3624.6 (Δ=+2.6)
2025-11-10 13:12:44,170 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=189 → R(idx=3038): 3634.0→3634.5 (Δ=+0.5) | G(idx=3038): 3634.0→3635.1 (Δ=+1.1) | B(idx=3038): 3634.0→3636.7 (Δ=+2.7)
2025-11-10 13:12:44,171 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=190 → R(idx=3058): 3647.0→3647.5 (Δ=+0.5) | G(idx=3058): 3647.0→3648.0 (Δ=+1.0) | B(idx=3058): 3647.0→3649.8 (Δ=+2.8)
2025-11-10 13:12:44,171 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=191 → R(idx=3058): 3647.0→3647.5 (Δ=+0.5) | G(idx=3058): 3647.0→3648.0 (Δ=+1.0) | B(idx=3058): 3647.0→3649.8 (Δ=+2.8)
2025-11-10 13:12:44,171 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=192 → R(idx=3078): 3661.0→3661.5 (Δ=+0.5) | G(idx=3078): 3661.0→3662.0 (Δ=+1.0) | B(idx=3078): 3661.0→3664.1 (Δ=+3.1)
2025-11-10 13:12:44,171 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=193 → R(idx=3098): 3669.0→3669.6 (Δ=+0.6) | G(idx=3098): 3669.0→3670.3 (Δ=+1.3) | B(idx=3098): 3669.0→3673.0 (Δ=+4.0)
2025-11-10 13:12:44,171 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=194 → R(idx=3114): 3675.0→3675.6 (Δ=+0.6) | G(idx=3114): 3675.0→3676.2 (Δ=+1.2) | B(idx=3114): 3675.0→3678.8 (Δ=+3.8)
2025-11-10 13:12:44,171 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=195 → R(idx=3134): 3682.0→3682.6 (Δ=+0.6) | G(idx=3134): 3682.0→3683.3 (Δ=+1.3) | B(idx=3134): 3682.0→3687.1 (Δ=+5.1)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=196 → R(idx=3154): 3690.0→3690.6 (Δ=+0.6) | G(idx=3154): 3690.0→3691.2 (Δ=+1.2) | B(idx=3154): 3690.0→3694.4 (Δ=+4.4)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=197 → R(idx=3171): 3696.0→3696.7 (Δ=+0.7) | G(idx=3171): 3696.0→3697.4 (Δ=+1.4) | B(idx=3171): 3696.0→3701.0 (Δ=+5.0)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=198 → R(idx=3171): 3696.0→3696.7 (Δ=+0.7) | G(idx=3171): 3696.0→3697.4 (Δ=+1.4) | B(idx=3171): 3696.0→3701.0 (Δ=+5.0)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=199 → R(idx=3191): 3703.0→3703.8 (Δ=+0.8) | G(idx=3191): 3703.0→3704.5 (Δ=+1.5) | B(idx=3191): 3703.0→3707.9 (Δ=+4.9)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=200 → R(idx=3211): 3711.0→3711.8 (Δ=+0.8) | G(idx=3211): 3711.0→3712.6 (Δ=+1.6) | B(idx=3211): 3711.0→3717.4 (Δ=+6.4)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=201 → R(idx=3227): 3717.0→3717.7 (Δ=+0.7) | G(idx=3227): 3717.0→3718.5 (Δ=+1.5) | B(idx=3227): 3717.0→3723.5 (Δ=+6.5)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=202 → R(idx=3247): 3724.0→3724.8 (Δ=+0.8) | G(idx=3247): 3724.0→3725.7 (Δ=+1.7) | B(idx=3247): 3724.0→3729.7 (Δ=+5.7)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=203 → R(idx=3267): 3731.0→3731.8 (Δ=+0.8) | G(idx=3267): 3731.0→3732.6 (Δ=+1.6) | B(idx=3267): 3731.0→3737.5 (Δ=+6.5)
2025-11-10 13:12:44,172 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=204 → R(idx=3283): 3737.0→3737.8 (Δ=+0.8) | G(idx=3283): 3737.0→3738.6 (Δ=+1.6) | B(idx=3283): 3737.0→3744.5 (Δ=+7.5)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=205 → R(idx=3303): 3745.0→3745.9 (Δ=+0.9) | G(idx=3303): 3745.0→3746.8 (Δ=+1.8) | B(idx=3303): 3745.0→3751.3 (Δ=+6.3)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=206 → R(idx=3303): 3745.0→3745.9 (Δ=+0.9) | G(idx=3303): 3745.0→3746.8 (Δ=+1.8) | B(idx=3303): 3745.0→3751.3 (Δ=+6.3)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=207 → R(idx=3324): 3752.0→3752.9 (Δ=+0.9) | G(idx=3324): 3752.0→3753.9 (Δ=+1.9) | B(idx=3324): 3752.0→3760.4 (Δ=+8.4)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=208 → R(idx=3340): 3758.0→3758.9 (Δ=+0.9) | G(idx=3340): 3758.0→3759.8 (Δ=+1.8) | B(idx=3340): 3758.0→3766.1 (Δ=+8.1)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=209 → R(idx=3360): 3765.0→3766.0 (Δ=+1.0) | G(idx=3360): 3765.0→3766.9 (Δ=+1.9) | B(idx=3360): 3765.0→3774.7 (Δ=+9.7)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=210 → R(idx=3380): 3773.0→3773.9 (Δ=+0.9) | G(idx=3380): 3773.0→3774.7 (Δ=+1.7) | B(idx=3380): 3773.0→3780.2 (Δ=+7.2)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=211 → R(idx=3400): 3780.0→3781.0 (Δ=+1.0) | G(idx=3400): 3780.0→3781.9 (Δ=+1.9) | B(idx=3400): 3780.0→3789.9 (Δ=+9.9)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=212 → R(idx=3416): 3786.0→3787.0 (Δ=+1.0) | G(idx=3416): 3786.0→3788.0 (Δ=+2.0) | B(idx=3416): 3786.0→3795.6 (Δ=+9.6)
2025-11-10 13:12:44,173 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=213 → R(idx=3416): 3786.0→3787.0 (Δ=+1.0) | G(idx=3416): 3786.0→3788.0 (Δ=+2.0) | B(idx=3416): 3786.0→3795.6 (Δ=+9.6)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=214 → R(idx=3436): 3793.0→3793.9 (Δ=+0.9) | G(idx=3436): 3793.0→3795.0 (Δ=+2.0) | B(idx=3436): 3793.0→3803.7 (Δ=+10.7)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=215 → R(idx=3457): 3801.0→3802.0 (Δ=+1.0) | G(idx=3457): 3801.0→3802.9 (Δ=+1.9) | B(idx=3457): 3801.0→3811.1 (Δ=+10.1)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=216 → R(idx=3473): 3807.0→3808.0 (Δ=+1.0) | G(idx=3473): 3807.0→3809.1 (Δ=+2.1) | B(idx=3473): 3807.0→3818.2 (Δ=+11.2)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=217 → R(idx=3493): 3814.0→3815.1 (Δ=+1.1) | G(idx=3493): 3814.0→3816.2 (Δ=+2.2) | B(idx=3493): 3814.0→3825.7 (Δ=+11.7)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=218 → R(idx=3513): 3822.0→3823.2 (Δ=+1.2) | G(idx=3513): 3822.0→3824.2 (Δ=+2.2) | B(idx=3513): 3822.0→3834.4 (Δ=+12.4)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=219 → R(idx=3529): 3828.0→3829.1 (Δ=+1.1) | G(idx=3529): 3828.0→3830.3 (Δ=+2.3) | B(idx=3529): 3828.0→3842.5 (Δ=+14.5)
2025-11-10 13:12:44,174 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=220 → R(idx=3529): 3828.0→3829.1 (Δ=+1.1) | G(idx=3529): 3828.0→3830.3 (Δ=+2.3) | B(idx=3529): 3828.0→3842.5 (Δ=+14.5)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=221 → R(idx=3549): 3835.0→3836.2 (Δ=+1.2) | G(idx=3549): 3835.0→3837.3 (Δ=+2.3) | B(idx=3549): 3835.0→3847.9 (Δ=+12.9)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=222 → R(idx=3569): 3842.0→3843.3 (Δ=+1.3) | G(idx=3569): 3842.0→3844.6 (Δ=+2.6) | B(idx=3569): 3842.0→3859.4 (Δ=+17.4)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=223 → R(idx=3585): 3848.0→3849.2 (Δ=+1.2) | G(idx=3585): 3848.0→3850.3 (Δ=+2.3) | B(idx=3585): 3848.0→3866.1 (Δ=+18.1)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=224 → R(idx=3606): 3856.0→3857.1 (Δ=+1.1) | G(idx=3606): 3856.0→3858.2 (Δ=+2.2) | B(idx=3606): 3856.0→3876.9 (Δ=+20.9)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=225 → R(idx=3626): 3866.0→3867.1 (Δ=+1.1) | G(idx=3626): 3866.0→3868.3 (Δ=+2.3) | B(idx=3626): 3866.0→3881.0 (Δ=+15.0)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=226 → R(idx=3642): 3874.0→3875.2 (Δ=+1.2) | G(idx=3642): 3874.0→3876.5 (Δ=+2.5) | B(idx=3642): 3874.0→3894.1 (Δ=+20.1)
2025-11-10 13:12:44,175 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=227 → R(idx=3642): 3874.0→3875.2 (Δ=+1.2) | G(idx=3642): 3874.0→3876.5 (Δ=+2.5) | B(idx=3642): 3874.0→3894.1 (Δ=+20.1)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=228 → R(idx=3662): 3883.0→3884.1 (Δ=+1.1) | G(idx=3662): 3883.0→3885.3 (Δ=+2.3) | B(idx=3662): 3883.0→3901.3 (Δ=+18.3)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=229 → R(idx=3682): 3893.0→3894.1 (Δ=+1.1) | G(idx=3682): 3893.0→3895.2 (Δ=+2.2) | B(idx=3682): 3893.0→3912.6 (Δ=+19.6)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=230 → R(idx=3698): 3901.0→3902.1 (Δ=+1.1) | G(idx=3698): 3901.0→3903.2 (Δ=+2.2) | B(idx=3698): 3901.0→3921.7 (Δ=+20.7)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=231 → R(idx=3718): 3911.0→3912.1 (Δ=+1.1) | G(idx=3718): 3911.0→3913.2 (Δ=+2.2) | B(idx=3718): 3911.0→3929.5 (Δ=+18.5)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=232 → R(idx=3739): 3921.0→3922.1 (Δ=+1.1) | G(idx=3739): 3921.0→3923.4 (Δ=+2.4) | B(idx=3739): 3921.0→3947.3 (Δ=+26.3)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=233 → R(idx=3755): 3929.0→3930.2 (Δ=+1.2) | G(idx=3755): 3929.0→3931.4 (Δ=+2.4) | B(idx=3755): 3929.0→3950.9 (Δ=+21.9)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=234 → R(idx=3755): 3929.0→3930.2 (Δ=+1.2) | G(idx=3755): 3929.0→3931.4 (Δ=+2.4) | B(idx=3755): 3929.0→3950.9 (Δ=+21.9)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=235 → R(idx=3775): 3939.0→3940.2 (Δ=+1.2) | G(idx=3775): 3939.0→3941.5 (Δ=+2.5) | B(idx=3775): 3939.0→3967.8 (Δ=+28.8)
2025-11-10 13:12:44,176 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=236 → R(idx=3795): 3948.0→3949.2 (Δ=+1.2) | G(idx=3795): 3948.0→3950.4 (Δ=+2.4) | B(idx=3795): 3948.0→3983.3 (Δ=+35.3)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=237 → R(idx=3811): 3956.0→3957.2 (Δ=+1.2) | G(idx=3811): 3956.0→3958.3 (Δ=+2.3) | B(idx=3811): 3956.0→3986.4 (Δ=+30.4)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=238 → R(idx=3831): 3966.0→3967.1 (Δ=+1.1) | G(idx=3831): 3966.0→3968.3 (Δ=+2.3) | B(idx=3831): 3966.0→3998.4 (Δ=+32.4)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=239 → R(idx=3852): 3976.0→3977.1 (Δ=+1.1) | G(idx=3852): 3976.0→3978.1 (Δ=+2.1) | B(idx=3852): 3976.0→4009.2 (Δ=+33.2)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=240 → R(idx=3868): 3984.0→3985.0 (Δ=+1.0) | G(idx=3868): 3984.0→3986.0 (Δ=+2.0) | B(idx=3868): 3984.0→4014.9 (Δ=+30.9)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=241 → R(idx=3888): 3994.0→3995.0 (Δ=+1.0) | G(idx=3888): 3994.0→3996.1 (Δ=+2.1) | B(idx=3888): 3994.0→4031.9 (Δ=+37.9)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=242 → R(idx=3888): 3994.0→3995.0 (Δ=+1.0) | G(idx=3888): 3994.0→3996.1 (Δ=+2.1) | B(idx=3888): 3994.0→4031.9 (Δ=+37.9)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=243 → R(idx=3908): 4004.0→4005.1 (Δ=+1.1) | G(idx=3908): 4004.0→4006.0 (Δ=+2.0) | B(idx=3908): 4004.0→4042.9 (Δ=+38.9)
2025-11-10 13:12:44,177 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=244 → R(idx=3924): 4012.0→4013.0 (Δ=+1.0) | G(idx=3924): 4012.0→4014.1 (Δ=+2.1) | B(idx=3924): 4012.0→4063.2 (Δ=+51.2)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=245 → R(idx=3944): 4021.0→4021.9 (Δ=+0.9) | G(idx=3944): 4021.0→4022.9 (Δ=+1.9) | B(idx=3944): 4021.0→4068.8 (Δ=+47.8)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=246 → R(idx=3964): 4031.0→4032.1 (Δ=+1.1) | G(idx=3964): 4031.0→4032.9 (Δ=+1.9) | B(idx=3964): 4031.0→4095.0 (Δ=+64.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=247 → R(idx=3980): 4039.0→4039.9 (Δ=+0.9) | G(idx=3980): 4039.0→4040.7 (Δ=+1.7) | B(idx=3980): 4039.0→4095.0 (Δ=+56.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=248 → R(idx=4000): 4049.0→4049.9 (Δ=+0.9) | G(idx=4000): 4049.0→4050.7 (Δ=+1.7) | B(idx=4000): 4049.0→4095.0 (Δ=+46.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=249 → R(idx=4000): 4049.0→4049.9 (Δ=+0.9) | G(idx=4000): 4049.0→4050.7 (Δ=+1.7) | B(idx=4000): 4049.0→4095.0 (Δ=+46.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=250 → R(idx=4016): 4057.0→4058.0 (Δ=+1.0) | G(idx=4016): 4057.0→4058.8 (Δ=+1.8) | B(idx=4016): 4057.0→4095.0 (Δ=+38.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=251 → R(idx=4036): 4067.0→4068.0 (Δ=+1.0) | G(idx=4036): 4067.0→4068.9 (Δ=+1.9) | B(idx=4036): 4067.0→4095.0 (Δ=+28.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=252 → R(idx=4056): 4077.0→4077.7 (Δ=+0.7) | G(idx=4056): 4077.0→4078.5 (Δ=+1.5) | B(idx=4056): 4077.0→4095.0 (Δ=+18.0)
2025-11-10 13:12:44,178 - INFO - subpage_vacspace.py:1404 - [Batch Correction] g=253 → R(idx=4072): 4085.0→4085.6 (Δ=+0.6) | G(idx=4072): 4085.0→4086.3 (Δ=+1.3) | B(idx=4072): 4085.0→4095.0 (Δ=+10.0)
d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py:1416: RuntimeWarning: invalid value encountered in cast
  new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)
이런식으로 나옵니다. 이걸 DF로 출력되도록 할 수 있나요?
또 RuntimeWarning: invalid value encountered in cast 경고는 왜 발생했고 어떻게 해결해야 하나요?
또 중복되는 기능의 Log가 많아서 Log는 {}회차 보정 결과:
gray | LUT idx | ΔR | ΔG | ΔB | R | G | B 
이런식으로 DF만 남기고 싶어요.
