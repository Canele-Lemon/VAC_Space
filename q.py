    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        """
        조건 1) spec_ok==True: 종료
        조건 2) (spec_ok==False) and (max_iters>0): Batch Correction
                └─ 조건 2-1) (len(ng_grays)<=10): Batch Correction 종료 → Per-gray Fine Correction 진입
                └─ 조건 2-2) (len(ng_grays)>10) and (iter_idx<max_iters): 다음 Batch Correction
        """
        try:
            ng_grays = []
            thr_g = None
            thr_c = None
            
            if metrics and "error" not in metrics:
                max_dG   = metrics.get("max_dG",  float("nan"))
                max_dCx  = metrics.get("max_dCx", float("nan"))
                max_dCy  = metrics.get("max_dCy", float("nan"))
                thr_g    = metrics.get("thr_gamma", self._spec_thread.thr_gamma if self._spec_thread else None)
                thr_c    = metrics.get("thr_c",     self._spec_thread.thr_c     if self._spec_thread else None)
                ng_grays = metrics.get("ng_grays", [])
                
                logging.info(
                    f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{thr_g}), "
                    f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c}), "
                    f"NG grays={ng_grays}"
                )
            else:
                logging.warning("[Evaluation] evaluation failed — treating as not passed.")
                ng_grays = []

            self._update_spec_views(iter_idx, self._off_store, self._on_store)

            # 조건 1) spec_ok==True: 종료
            if spec_ok:
                self._step_done(5)
                logging.info("[Evaluation] Spec 통과 — 최적화 종료")
                return
            
            self._step_fail(5)
            if max_iters <= 0:
                logging.info("[Evaluation] Spec NG but no further correction (max_iters≤0) - 최적화 종료")
                return
            
            # 조건 2) (spec_ok==False) and (max_iters>0): Batch Correction
            ng_cnt = len(ng_grays)
            # 조건 2-1) (len(ng_grays)<=10): Batch Correction 종료 → Per-gray Fine Correction 진입
            if ng_cnt > 0 and ng_cnt <= 10:
                logging.info(f"[Evaluation] NG gray {ng_cnt}개 ≤ 10 → Batch Correction 종료, Per-gray Fine Correction 시작")
                for s in (2, 3, 4):
                    self._step_set_pending(s)
                thr_gamma = float(thr_g) if thr_g is not None else 0.05
                thr_c_val = float(thr_c) if thr_c is not None else 0.003
                self._start_fine_correction_for_ng_list(
                    ng_grays,
                    thr_gamma=thr_gamma,
                    thr_c=thr_c_val
                )
                return
            # 조건 2-2) (len(ng_grays)>10) and (iter_idx<max_iters): 다음 Batch Correction
            if iter_idx < max_iters:
                logging.info(f"[Evaluation] Spec NG — batch 보정 {iter_idx+1}회차 시작")
                for s in (2, 3, 4):
                    self._step_set_pending(s)

                thr_gamma = float(thr_g) if thr_g is not None else 0.05
                thr_c_val = float(thr_c) if thr_c is not None else 0.003

                self._run_batch_correction_with_jacobian(
                    iter_idx=iter_idx+1,
                    max_iters=max_iters,
                    thr_gamma=thr_gamma,
                    thr_c=thr_c_val,
                    metrics=metrics
                )
            else:
                logging.info("[Correction] 최대 보정 횟수 도달 — 종료")

        finally:
            self._spec_thread = None

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
            dX = self._solve_delta_rgb_for_gray(g, d_targets, lam=lam,
                                                wCx=0.5, wCy=0.5, wG=1.0)
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
        wCx: float = 0.5,
        wCy: float = 0.5,
        wG:  float = 1.0,
    ):
        """
        주어진 gray g에서, 현재 ΔY = [dCx, dCy, dGamma]를
        자코비안 J_g를 이용해 줄이기 위한 ΔX = [ΔR_H, ΔG_H, ΔB_H]를 푼다.

        관계식:  ΔY_new ≈ ΔY + J_g · ΔX
        우리가 원하는 건 ΔY_new ≈ 0 이므로, J_g · ΔX ≈ -ΔY 를 풀어야 함.

        리지 가중 최소자승:
            argmin_ΔX || W (J_g ΔX + ΔY) ||^2 + λ ||ΔX||^2
            → (J^T W^2 J + λI) ΔX = - J^T W^2 ΔY
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

        # 가중치
        w_vec = np.array([wCx, wCy, wG], dtype=np.float32)     # (3,)
        WJ = w_vec[:, None] * Jg   # (3,3)
        Wy = w_vec * dy            # (3,)

        A = WJ.T @ WJ + float(lam) * np.eye(3, dtype=np.float32)  # (3,3)
        b = - WJ.T @ Wy                                           # (3,)

        try:
            dX = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)
        
        step_gain = 32.0
        dR, dG, dB = (float(dX[0]) * step_gain, 
                      float(dX[1]) * step_gain, 
                      float(dX[2]) * step_gain)
        logging.debug(
            f"[BATCH CORR] g={g}: dCx={dCx_g:+.6f}, dCy={dCy_g:+.6f}, dG={dG_g:+.6f} → "
            f"ΔR_H={dR:+.3f}, ΔG_H={dG:+.3f}, ΔB_H={dB:+.3f}"
        )
        return dR, dG, dB

네. 현재 보정 코드 드립니다. NG인 component에 따라 가중치를 줄 수 있도록 코드를 어떻게 수정하면 되는지 알려주세요
