    # =======================================================================================================
    # 2. VAC Optimization Workflow
    # =======================================================================================================
    def start_vac_optimization(self):
        if not self._check_vac_optimization_validation():
            return

        if not self._confirm_vac_optimization_target():
            return

        self._reset_vac_optimization_ui_state()
        
        self._spec_policy = VACSpecPolicy()
        
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        self._fine_mode = False
        self._fine_ng_list = None
        
        self._load_jacobian_bundle_npy()
        self._load_prediction_models()
        
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")
        
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self.measure_off_ref_then_on()

    def measure_off_ref_then_on(self):
        profile_off = SessionProfile(
            session_mode="VAC OFF",
            cie_label="data_1",
            table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
            ref_store=None
        )

        def _after_off(store_off):
            self._off_store = store_off
            lv_off = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = store_off['gamma']['main']['white'].get(g, None)
                lv_off[g] = float(tup[0]) if tup else np.nan
            self._gamma_off_vec = self.compute_gamma_series(lv_off)
            
            self._lv_off_vec = lv_off.copy()
            try:
                self._lv_off_max = float(np.nanmax(lv_off[1:]))
            except (ValueError, TypeError):
                self._lv_off_max = float('nan')
            
            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
            logging.info("[TV Control] VAC ON 전환 성공")
            
            logging.info("[Measurement] VAC ON 측정 시작")
            self.apply_predicted_vac_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off,
            on_done=_after_off
        )

    def apply_predicted_vac_and_measure_on(self):
        self._step_start(2)
        
        BASE_VAC_PK = 3025
        vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(BASE_VAC_PK)
        if base_vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        base_vac_dict = json.loads(base_vac_data)
        self._vac_dict_cache = base_vac_dict
        
        try:
            predicted_vac_data, new_lut_4096, debug_info = self._generate_predicted_vac_lut(
                base_vac_dict,
                n_iters=1,
                wG=0.4,
                wC=1.0,
                lambda_ridge=1e-3
            )
            if predicted_vac_data is None:
                raise RuntimeError("predicted_vac_data is None")
        except Exception:
            logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
            predicted_vac_data = base_vac_data
            debug_info = None
            
        predicted_vac_dict = json.loads(predicted_vac_data)
        self._vac_dict_cache = predicted_vac_dict
            
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in predicted_vac_dict.items() if "channel" in key}
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            self.send_command(self.ser_tv, 'exit')
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self.verify_vac_data_match(written_data=predicted_vac_dict, read_data=read_vac_dict)

            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")

            self._step_done(3)

            self._fine_mode = False
            
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_on = SessionProfile(
                session_mode="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                pol = self._spec_policy
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self.on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=1))
                self._spec_thread.start()

            logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 시작")
            
            self._step_start(4)
            self._step_set_pending(5)

            self.start_viewing_angle_session(
                profile=profile_on,
                on_done=_after_on
            )

        logging.info("[VAC Writing] 예측기반 최적화 VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(predicted_vac_data, on_finished=_after_write)

    def on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        """
        조건 1) spec_ok==True: 종료
        조건 2) (spec_ok==False) and (iter_idx < max_iters): NG Gray batch correction 반복
        """
        try:
            pol = self._spec_policy
            
            # logging
            ng_grays = []
            if metrics and "error" not in metrics:
                max_dG  = metrics.get("max_dG",  float("nan"))
                max_dCx = metrics.get("max_dCx", float("nan"))
                max_dCy = metrics.get("max_dCy", float("nan"))
                ng_grays = metrics.get("ng_grays", [])
                
                logging.info(
                    f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{pol.thr_gamma}), "
                    f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{pol.thr_c}), "
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
                
                try:
                    self._final_vac_data_for_download = self._build_vacparam_std_format(
                        base_vac_dict=self._vac_dict_cache,
                        new_lut_tvkeys=None
                    )
                except Exception:
                    logging.exception("[Download] final vac data build failed")
                    self._final_vac_data_for_download = None
                    
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            # 조건 2) (spec_ok==False) and (max_iters>0): NG Gray Correction
            self._step_fail(5)
            
            if max_iters <= 0:
                logging.info("[Evaluation] Spec NG but no further correction (max_iters≤0) - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            if iter_idx >= max_iters:
                logging.info("[Evaluation] Spec NG but 보정 횟수 초과 - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            for s in (2, 3, 4):
                self._step_set_pending(s)
                
            self.run_batch_correction_with_jacobian(
                iter_idx=iter_idx+1,
                max_iters=max_iters,
                policy=pol,
                metrics=metrics
            )
            
        finally:
            self._spec_thread = None

    def run_batch_correction_with_jacobian(self, iter_idx, max_iters, policy: VACSpecPolicy, lam=1e-3, metrics=None):
        logging.info(f"[Batch Correction] iteration {iter_idx} start (Jacobian dense)")

        self._step_start(2)

        # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
        if not hasattr(self, "_J_dense"):
            logging.error("[Batch Correction] J_dense not loaded")
            return
        
        self._load_mapping_index_gray_to_lut()
        
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.error("[Batch Correction] no VAC cache; need latest TV VAC JSON")
            return

        # 1) NG gray 리스트 / Δ 타깃 준비            
        if metrics is not None and ("ng_grays" in metrics) and ("dG" in metrics) and ("dCx" in metrics) and ("dCy" in metrics):
            ng_list = list(metrics["ng_grays"])
            d_targets = {
                "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
                "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
                "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
            }
            logging.info(f"[Batch Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
        
        else:
            dG, dCx, dCy, ng_list, *_ = SpecEvalThread.compute_gray_errors_and_ng_list(
                self._off_store, self._on_store, policy
            )
            d_targets = {
                "Gamma": dG.astype(np.float32),
                "Cx":    dCx.astype(np.float32),
                "Cy":    dCy.astype(np.float32),
            }
            logging.info(f"[Batch Correction] NG grays (recomputed by policy): {ng_list}")

        if not ng_list:
            logging.info("[Batch Correction] no NG gray → 보정 없음")
            return
    
        # 2) 현재 High LUT 확보
        vac_dict = self._vac_dict_cache
        RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
        GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
        BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

        RH = RH0.copy()
        GH = GH0.copy()
        BH = BH0.copy()

        # 3) index별 Δ 누적
        delta_acc = {"R": np.zeros_like(RH), "G": np.zeros_like(GH), "B": np.zeros_like(BH)}
        count_acc = {"R": np.zeros_like(RH, dtype=np.int32),
                    "G": np.zeros_like(GH, dtype=np.int32),
                    "B": np.zeros_like(BH, dtype=np.int32)}

        mapLUT = self._mapping_index_gray_to_lut
        
        n_gray = 256
        dR_gray = np.full(n_gray, np.nan, np.float32)
        dG_gray = np.full(n_gray, np.nan, np.float32)
        dB_gray = np.full(n_gray, np.nan, np.float32)
        corr_flag = np.zeros(n_gray, np.int32)
        wCx_gray = np.full(n_gray, np.nan, np.float32)
        wCy_gray = np.full(n_gray, np.nan, np.float32)
        wG_gray  = np.full(n_gray, np.nan, np.float32)
        
        step_gain_last = 1.0
        
        # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
        for g in ng_list:
            if 0 <= g < n_gray:
                corr_flag[g] = 1
                
            dX = self._solve_delta_rgb_for_gray(
                g,
                d_targets,
                lam=lam,
                thr_c=policy.thr_c,             # 색좌표 스펙
                thr_gamma=policy.thr_gamma,     # 감마 스펙
                base_wCx=0.5,                   # Cx 기본 가중치 (기존 0.5를 base로 사용)
                base_wCy=0.5,                   # Cy 기본 가중치
                base_wG=1.0,                    # Gamma 기본 가중치
                boost=3.0,                      # NG일 때 배율
                keep=0.2,                       # OK일 때 배율
            )
            if dX is None:
                continue

            dR, dG, dB, wCx_g, wCy_g, wG_g, step_gain = dX
            step_gain_last = step_gain
            
            dR_gray[g] = dR
            dG_gray[g] = dG
            dB_gray[g] = dB
            wCx_gray[g] = wCx_g
            wCy_gray[g] = wCy_g
            wG_gray[g]  = wG_g

            idx = int(mapLUT[g])
            if 0 <= idx < len(RH):
                delta_acc["R"][idx] += dR
                count_acc["R"][idx] += 1
            if 0 <= idx < len(GH):
                delta_acc["G"][idx] += dG
                count_acc["G"][idx] += 1
            if 0 <= idx < len(BH):
                delta_acc["B"][idx] += dB
                count_acc["B"][idx] += 1

        # 5) index별 평균 Δ 적용 + clip + monotone + 로그
        for ch, arr, arr0 in (("R", RH, RH0), ("G", GH, GH0), ("B", BH, BH0)):
            da = delta_acc[ch]
            ct = count_acc[ch]
            mask = ct > 0
            if not np.any(mask):
                logging.info(f"[Batch Correction] channel {ch}: no indices updated")
                continue
            arr[mask] = arr0[mask] + (da[mask] / ct[mask])  # 평균 Δ
            arr[:] = np.clip(arr, 0.0, 4095.0)              # clip
            self.enforce_monotone(arr)                      # 단조 증가 (i<j → LUT[i] ≤ LUT[j])

        # 6) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "RchannelHigh": RH,
            "GchannelHigh": GH,
            "BchannelHigh": BH,
        }
        for k in new_lut_4096:            
            arr = np.asarray(new_lut_4096[k], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            new_lut_4096[k] = np.clip(np.round(arr), 0, 4095).astype(np.uint16)
        
        # 7) 보정 결과 로그/저장/시각화
        df_corr = self._build_batch_corr_df(
            iter_idx=iter_idx,
            d_targets=d_targets,
            dR_gray=dR_gray, dG_gray=dG_gray, dB_gray=dB_gray,
            corr_flag=corr_flag,
            mapLUT=mapLUT,
            RH0=RH0, GH0=GH0, BH0=BH0,
            RH=RH, GH=GH, BH=BH,
            wCx_gray=wCx_gray, wCy_gray=wCy_gray, wG_gray=wG_gray,
        )
        logging.info(
            f"[Batch Correction] {iter_idx}회차 보정 결과:\n"
            + df_corr.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )
        self._save_batch_corr_df(iter_idx, df_corr, step_gain=step_gain_last)

        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
        logging.info(f"[VAC Writing] LUT {iter_idx}차 보정 VAC Data TV Writing start")

        vac_corr_data = self._build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )
        vac_corr_dict = json.loads(vac_corr_data)
        self._vac_dict_cache = vac_corr_dict

        def _after_write(ok, msg):
            logging.info(f"[VAC Writing] write result: {ok} {msg}")
            if not ok:
                return
            logging.info("[VAC Reading] TV reading after write")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            self.send_command(self.ser_tv, 'exit')
            if not vac_dict_after:
                logging.error("[VAC Reading] TV read-back failed")
                return
            
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self.verify_vac_data_match(written_data=vac_corr_dict, read_data=vac_dict_after)
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
                session_mode=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr
                self._update_last_on_lv_norm(store_corr)
                
                self._step_start(5)
                pol = self._spec_policy
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
                self._spec_thread.finished.connect(lambda ok, m: self.on_spec_eval_done(ok, m, iter_idx, max_iters))
                self._spec_thread.start()

            logging.info(f"[Measurement] LUT {iter_idx}차 보정 기준 re-measure start")
            
            self._step_start(4)
            self._step_set_pending(5)
            
            self.start_viewing_angle_session(
                profile=profile_corr,
                on_done=_after_corr
            )

        self._step_start(3)
        self._write_vac_to_tv(vac_corr_data, on_finished=_after_write)

    # =====================================================================================================
    # 3. VAC Apply / Re-evaluation / Failover Workflow
    # =====================================================================================================
    def apply_vac_by_pk_and_re_evaluate(self, vac_info_pk: int, thr_gamma: float, thr_c: float):
        try:
            self._step_start(2)

            logging.info(f"[DB] VAC(pk={vac_info_pk}) 로딩 시작")
            vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(vac_info_pk)
            if base_vac_data is None:
                logging.error(f"[DB] VAC(pk={vac_info_pk}) 로딩 실패 — 대체 적용 중단")
                self._step_fail(2)
                return

            base_vac_dict = json.loads(base_vac_data)
            self._vac_dict_cache = base_vac_dict

            try:
                lut_dict_plot = {key.replace("channel", "_"): v for key, v in base_vac_dict.items() if "channel" in key}
                self._update_lut_chart_and_table(lut_dict_plot)
            except Exception:
                logging.exception("[UI] LUT 차트/테이블 갱신 중 예외(계속 진행)")

            self._step_done(2)

            def _after_write(ok, msg):
                if not ok:
                    logging.error(f"[VAC Writing] VAC(pk={vac_info_pk}) Writing 실패: {msg}")
                    return
                logging.info(f"[VAC Writing] VAC(pk={vac_info_pk}) Writing 완료: {msg}")

                self._last_written_base_vac_dict = base_vac_dict
                self._last_written_new_lut_tvkeys = None

                logging.info("[VAC Reading] 시작")
                self._read_vac_from_tv(_after_read)

            def _after_read(read_vac_dict):
                self.send_command(self.ser_tv, 'exit')
                if not read_vac_dict:
                    logging.error("[VAC Reading] 실패 — 재평가 중단")
                    return

                mismatch_keys = self.verify_vac_data_match(written_data=base_vac_dict, read_data=read_vac_dict)
                if mismatch_keys:
                    logging.warning(f"[VAC Reading] 데이터 불일치 — keys={mismatch_keys} — 재평가 중단")
                    return
                else:
                    logging.info("[VAC Reading] Written/Read VAC 일치")

                self._step_done(3)

                # ON 측정 & 재평가
                try:
                    self._fine_mode = False
                    self.vac_optimization_gamma_chart.reset_on()
                    self.vac_optimization_cie1976_chart.reset_on()
                except Exception:
                    logging.exception("[UI] 차트 reset 중 예외(계속 진행)")

                profile_on = SessionProfile(
                    session_mode="VAC ON",
                    cie_label="data_2",
                    table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                    ref_store=self._off_store
                )

                def _after_on(store_on):
                    logging.info("[Measurement] VAC(pk=%s) 기준 ON 측정 완료", vac_info_pk)
                    self._step_done(4)
                    self._on_store = store_on
                    self._update_last_on_lv_norm(store_on)

                    logging.info("[Evaluation] ΔCx/ΔCy/ΔGamma 재평가 시작")
                    self._step_start(5)

                    # 재평가만 수행(보정 반복 없이)
                    self._spec_thread = SpecEvalThread(
                        self._off_store, self._on_store,
                        thr_gamma=thr_gamma, thr_c=thr_c, parent=self
                    )
                    # 재평가 결과는 다시 on_spec_eval_done으로 (max_iters=0 → 보정 없음)
                    self._spec_thread.finished.connect(
                        lambda ok, met: self.on_spec_eval_done(ok, met, iter_idx=0, max_iters=0)
                    )
                    self._spec_thread.start()

                logging.info("[Measurement] VAC(pk=%s) 기준 ON 측정 시작", vac_info_pk)
                self._step_start(4)
                self.start_viewing_angle_session(profile=profile_on, on_done=_after_on)

            logging.info("[VAC Writing] VAC(pk=%s) TV Writing 시작", vac_info_pk)
            self._write_vac_to_tv(base_vac_data, on_finished=_after_write)

        except Exception:
            logging.exception(f"[Failover] VAC(pk={vac_info_pk}) 적용/재평가 중 예외 발생")

    # =====================================================================================================
    # 4. Prediction / Jacobian Core
    # =====================================================================================================
    def _generate_predicted_vac_lut(
        self,
        base_vac_dict: dict,
        *,
        n_iters: int = 1,
        wG: float = 0.4, # dGamma weight
        wC: float = 1.0, # dCx/dCy weight
        lambda_ridge: float = 1e-3,
        use_pattern_onehot: bool = False,
        patterns: tuple = ("W",),
        bypass_vac_info_pk: int = 1,
    ):

        debug_info = {
            "iters": [],
            "bypass_vac_info_pk": bypass_vac_info_pk,
        }

        try:
            # 0) prerequisite check
            if not hasattr(self, "_J_dense") or self._J_dense is None:
                raise RuntimeError("[PredictOpt] Jacobian bundle (_J_dense) not loaded.")

            if not hasattr(self, "models_Y0_bundle") or self.models_Y0_bundle is None:
                raise RuntimeError("[PredictOpt] Prediction models (models_Y0_bundle) not loaded.")

            # 1) mapping index (gray->lut j)
            self._load_mapping_index_gray_to_lut()
            idx_map = np.asarray(self._mapping_index_gray_to_lut, dtype=np.int32)  # (256,)
            if idx_map.shape[0] != 256:
                raise ValueError(f"[PredictOpt] idx_map must be (256,), got {idx_map.shape}")

            # 2) load bypass VAC LUT (4096) from DB (pk=1)
            vac_version_b, bypass_vac_data = self._fetch_vac_by_vac_info_pk(bypass_vac_info_pk)
            if bypass_vac_data is None:
                raise RuntimeError(f"[PredictOpt] bypass VAC fetch failed. pk={bypass_vac_info_pk}")

            bypass_vac_dict = json.loads(bypass_vac_data)

            # 3) extract 4096 LUT arrays (base & bypass)
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

            # 4) 256 LUT @ mapped indices
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

            high_R = base_256["R_High"].copy()
            high_G = base_256["G_High"].copy()
            high_B = base_256["B_High"].copy()
            low_R = base_256["R_Low"].copy()
            low_G = base_256["G_Low"].copy()
            low_B = base_256["B_Low"].copy()

            # 5) meta (panel onehot + fr + model_year)
            artifact_panel, artifact_hz = self._resolve_artifact_key()

            panel_text = artifact_panel
            frame_rate = float(artifact_hz)

            panel_onehot = self.panel_text_to_onehot(panel_text).astype(np.float32)

            logging.debug(
                f"[Predict META] panel_text={panel_text}, frame_rate={frame_rate}, "
                f"panel_onehot_dim={len(panel_onehot)}"
            )

            pattern_order = list(patterns)
            def _pattern_onehot(p: str) -> np.ndarray:
                v = np.zeros(len(pattern_order), dtype=np.float32)
                if p in pattern_order:
                    v[pattern_order.index(p)] = 1.0
                return v

            # 6) helper: build X for model (per-gray)
            def _build_X_y0_per_gray(d_lut_256: dict, pat: str = "W") -> np.ndarray:
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
                    row.extend(panel_onehot.tolist())
                    row.append(float(frame_rate))
                    row.append(float(g / 255.0))
                    row.append(float(idx_map[g]))

                    if use_pattern_onehot:
                        row.extend(_pattern_onehot(pat).tolist())

                    X_rows.append(row)

                return np.asarray(X_rows, dtype=np.float32)  # (256, D)

            # 7) helper: ML predict dCx/dCy/dGamma (per-gray)
            def _predict_y0(d_lut_256: dict, pat: str = "W"):
                X = _build_X_y0_per_gray(d_lut_256, pat=pat)

                try:
                    expected = self.models_Y0_bundle["dCx"]["linear_model"].named_steps["scaler"].n_features_in_
                    logging.debug(f"[Predict X] X.shape={X.shape}, expected_features={expected}")
                except Exception:
                    logging.exception("[Predict X] failed to check expected feature count")

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

            # 8) ML prediction + Jacobian correction
            for it in range(1, n_iters + 1):
                d_lut_256 = {
                    "R_Low":  low_R  - bp_256["R_Low"],
                    "G_Low":  low_G  - bp_256["G_Low"],
                    "B_Low":  low_B  - bp_256["B_Low"],
                    "R_High": high_R - bp_256["R_High"],
                    "G_High": high_G - bp_256["G_High"],
                    "B_High": high_B - bp_256["B_High"],
                }

                pat = pattern_order[0] if pattern_order else "W"
                dCx_pred, dCy_pred, dGamma_pred = _predict_y0(d_lut_256, pat=pat)

                dh_R = np.zeros(256, dtype=np.float32)
                dh_G = np.zeros(256, dtype=np.float32)
                dh_B = np.zeros(256, dtype=np.float32)

                w_vec = np.array([wC, wC, wG], dtype=np.float32)

                for g in range(256):
                    Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3, 3)

                    if not np.isfinite(Jg).all():
                        continue

                    dy = np.array([
                        float(dCx_pred[g]),
                        float(dCy_pred[g]),
                        float(dGamma_pred[g]),
                    ], dtype=np.float32)

                    if not np.isfinite(dy).all():
                        continue

                    if np.all(np.abs(dy) < 1e-8):
                        continue

                    WJ = w_vec[:, None] * Jg
                    Wy = w_vec * dy

                    A = WJ.T @ WJ + float(lambda_ridge) * np.eye(3, dtype=np.float32)
                    b = -WJ.T @ Wy

                    try:
                        dRGB = np.linalg.solve(A, b).astype(np.float32)
                    except np.linalg.LinAlgError:
                        dRGB = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

                    dh_R[g] = dRGB[0]
                    dh_G[g] = dRGB[1]
                    dh_B[g] = dRGB[2]

                high_R = high_R + dh_R
                high_G = high_G + dh_G
                high_B = high_B + dh_B
                high_R = np.clip(self.enforce_monotone(high_R), 0, 4095)
                high_G = np.clip(self.enforce_monotone(high_G), 0, 4095)
                high_B = np.clip(self.enforce_monotone(high_B), 0, 4095)

                debug_info["iters"].append({
                    "iter": it,
                    "mode": "ml_prediction_then_jacobian",
                    "pred_summary": {
                        "dCx_mean": float(np.nanmean(dCx_pred)),
                        "dCy_mean": float(np.nanmean(dCy_pred)),
                        "dGamma_mean": float(np.nanmean(dGamma_pred)),
                        "dCx_abs_mean": float(np.nanmean(np.abs(dCx_pred))),
                        "dCy_abs_mean": float(np.nanmean(np.abs(dCy_pred))),
                        "dGamma_abs_mean": float(np.nanmean(np.abs(dGamma_pred))),
                        "dCx_max_abs": float(np.nanmax(np.abs(dCx_pred))),
                        "dCy_max_abs": float(np.nanmax(np.abs(dCy_pred))),
                        "dGamma_max_abs": float(np.nanmax(np.abs(dGamma_pred))),
                    },
                    "dh_summary": {
                        "dR_abs_mean": float(np.nanmean(np.abs(dh_R))),
                        "dG_abs_mean": float(np.nanmean(np.abs(dh_G))),
                        "dB_abs_mean": float(np.nanmean(np.abs(dh_B))),
                        "dR_max_abs": float(np.nanmax(np.abs(dh_R))),
                        "dG_max_abs": float(np.nanmax(np.abs(dh_G))),
                        "dB_max_abs": float(np.nanmax(np.abs(dh_B))),
                    }
                })

                logging.info(
                    f"[PredictOpt] iter {it}/{n_iters} prediction correction done. "
                    f"wG={wG}, wC={wC}, lam={lambda_ridge}"
                )

            # 9) 256 -> 4096 upsample (High only)
            new_lut_tvkeys = {
                "RchannelLow":  self.to_int_list4096(base_RL),  # base_RL: (4096,)
                "GchannelLow":  self.to_int_list4096(base_GL),
                "BchannelLow":  self.to_int_list4096(base_BL),
                "RchannelHigh": self.to_int_list4096(np.round(self.up256_to_4096(high_R))),
                "GchannelHigh": self.to_int_list4096(np.round(self.up256_to_4096(high_G))),
                "BchannelHigh": self.to_int_list4096(np.round(self.up256_to_4096(high_B))),
            }

            # 10) build json (TV write format)
            predicted_vac_data = self._build_vacparam_std_format(
                base_vac_dict=base_vac_dict,
                new_lut_tvkeys=new_lut_tvkeys
            )

            return predicted_vac_data, new_lut_tvkeys, debug_info

        except Exception:
            logging.exception("[PredictOpt] failed")
            return None, None, debug_info

    def _solve_delta_rgb_for_gray(
        self,
        g: int,
        d_targets: dict,
        lam: float = 1e-3,
        wCx: float | None = None,
        wCy: float | None = None,
        wG:  float | None = None,
        thr_c: float | None = None,
        thr_gamma: float | None = None,
        base_wCx: float = 1.0,
        base_wCy: float = 1.0,
        base_wG:  float = 1.0,
        boost: float = 3.0,
        keep: float = 0.2,
    ):
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
        
        if np.all(np.abs(dy) < 1e-6):
            return None

        # 1) 가중치 계산
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
            wCx_eff, wCy_eff, wG_eff = float(wCx), float(wCy), float(wG)

        else:
            wCx_eff, wCy_eff, wG_eff = base_wCx, base_wCy, base_wG

        w_vec = np.array([wCx_eff, wCy_eff, wG_eff], dtype=np.float32)

        # 2) 가중 least squares
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

    def _stack_basis(self, knots, L=256):
        knots = np.asarray(knots, dtype=np.int32)
        
        def _phi(g):
            K = len(knots)
            w = np.zeros(K, dtype=np.float32)
            if g <= knots[0]:
                w[0]=1.; return w
            if g >= knots[-1]:
                w[-1]=1.; return w
            i = np.searchsorted(knots, g) - 1
            g0, g1 = knots[i], knots[i+1]
            t = (g - g0) / max(1, (g1 - g0))
            w[i] = 1-t; w[i+1] = t
            return w
        return np.vstack([_phi(g) for g in range(L)])



