예측모델: VAC input에 따른 Cx, Cy, Gamma 값과 VAC OFF일 때의 Cx, Cy, Gamma 값과의 차이인 dCx, dCy, dGamma을 학습함
자코비안: Base VAC 기준 +50~-50 sweep을 주었을 때 Cx, Cy, Gamma 값들의 변화. 각 gray level에서 VAC 변화에 따른 Cx, Cy, Gamma 변화량

-VAC 최적화 로직 flow-
1. VAC OFF 측정
2. DB에서 Base VAC 불러온 후 예측모델을 이용해 VAC OFF 대비 dCx, dCy, dGamma 예측 후 이를 스펙 in으로 만들기 위해 자코비안 보정하여 예측 VAC를 generate함. (|dCx/dCy|<=0.003, |Gamma|<=0.05)
3. 예측 VAC를 TV에 적용 후 측정.
4. 각 NG gray에서 스펙 in을 만들기 위해 자코비안을 이용해 미세 보정을 하고 보정한 VAC를 TV 적용 후 해당 Gray만 측정
5. NG gray가 없어질때까지 보정-TV적용-측정 반복

이 플로우를 구현하고자 합니다. 현재까지 작성된 코드가 아래와 같을 때, 어떻게 수정하면 되나요?

    def start_VAC_optimization(self):
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
        self._measure_off_ref_then_on()

    def _measure_off_ref_then_on(self):
        profile_off = SessionProfile(
            legend_text="VAC OFF (Ref.)",
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
            self._gamma_off_vec = self._compute_gamma_series(lv_off)
            
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
            self._apply_predicted_vac_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off,
            on_done=_after_off
        )

    def _apply_predicted_vac_and_measure_on(self):
        self._step_start(2)

        vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(2)
        if base_vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        base_vac_dict = json.loads(base_vac_data)
        self._vac_dict_cache = base_vac_dict

        try:
            predicted_vac_data, debug_info = self._generate_predicted_vac_lut(
                base_vac_dict,
                n_iters=2,
                wG=0.4,
                wC=1.0,
                lambda_ridge=1e-3
            )
        except Exception:
            logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
            predicted_vac_json, debug_info = None, None
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] DB fetch VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] DB fetch VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'exit')
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=vac_dict, read_data=read_vac_dict)

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
                legend_text="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=0))
                self._spec_thread.start()

            logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                gamma_settle_ms=1000,
                cs_settle_ms=1000,
                on_done=_after_on
            )

        logging.info("[VAC Writing] DB fetch VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(predicted_vac_data, on_finished=_after_write)
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
    def _generate_predicted_vac_lut(self, vac_dict, *, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3):
        try:
            # 4096→256 다운샘플 (12bit 값 그대로)
            lut256 = {
                "R_Low":  self._down4096_to_256_float(vac_dict["RchannelLow"]),
                "R_High": self._down4096_to_256_float(vac_dict["RchannelHigh"]),
                "G_Low":  self._down4096_to_256_float(vac_dict["GchannelLow"]),
                "G_High": self._down4096_to_256_float(vac_dict["GchannelHigh"]),
                "B_Low":  self._down4096_to_256_float(vac_dict["BchannelLow"]),
                "B_High": self._down4096_to_256_float(vac_dict["BchannelHigh"]),
            }

            if not hasattr(self, "A_Gamma"):
                logging.error("[PredictOpt] Jacobian not prepared.")
                return None, None

            panel, fr, model_year = self._get_ui_meta()

            K   = len(self._jac_artifacts["knots"])
            Phi = self._stack_basis(self._jac_artifacts["knots"])

            # 이 변수들은 계속 12bit 스케일 유지
            high_R = lut256["R_High"].copy()
            high_G = lut256["G_High"].copy()
            high_B = lut256["B_High"].copy()

            for it in range(1, n_iters + 1):
                # ✅ 2️⃣ 여기서 예측에 넘길 때만 0~1 스케일로 정규화
                lut256_for_pred = {
                    k: np.asarray(v, np.float32) / 4095.0 for k, v in {
                        "R_Low": lut256["R_Low"],
                        "G_Low": lut256["G_Low"],
                        "B_Low": lut256["B_Low"],
                        "R_High": high_R,
                        "G_High": high_G,
                        "B_High": high_B,
                    }.items()
                }

                # ✅ 예측은 정규화된 LUT 사용
                y_pred = self._predict_Y0W_from_models(
                    lut256_for_pred,
                    panel_text=panel, frame_rate=fr, model_year=model_year
                )

                # (선택) 디버그용 CSV 저장
                self._debug_dump_predicted_Y0W(
                    y_pred, tag=f"iter{it}_{panel}_fr{int(fr)}_my{int(model_year)%100:02d}", save_csv=True
                )

                # 이후 부분은 기존 그대로 유지
                d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)
                A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
                b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)
                mask  = np.isfinite(b_cat)
                A_use = A_cat[mask,:]; b_use = b_cat[mask]
                ATA = A_use.T @ A_use
                rhs = A_use.T @ b_use
                ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
                delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

                dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
                corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

                # ✅ 보정은 12bit 스케일에서 수행
                high_R = np.clip(self._enforce_monotone(high_R + corr_R), 0, 4095)
                high_G = np.clip(self._enforce_monotone(high_G + corr_G), 0, 4095)
                high_B = np.clip(self._enforce_monotone(high_B + corr_B), 0, 4095)

                logging.info(f"[PredictOpt] iter {it} done. (wG={wG}, wC={wC})")

            # 6) 256→4096 업샘플 (Low는 그대로, High만 갱신)
            new_lut_4096 = {
                "RchannelLow":  np.asarray(vac_dict["RchannelLow"], dtype=np.float32),
                "GchannelLow":  np.asarray(vac_dict["GchannelLow"], dtype=np.float32),
                "BchannelLow":  np.asarray(vac_dict["BchannelLow"], dtype=np.float32),
                "RchannelHigh": self._up256_to_4096(high_R),
                "GchannelHigh": self._up256_to_4096(high_G),
                "BchannelHigh": self._up256_to_4096(high_B),
            }
            for k in new_lut_4096:
                new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

            # 7) UI 바로 업데이트 (차트+테이블)
            lut_plot = {
                "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
                "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
                "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
            }
            time.sleep(5)
            self._update_lut_chart_and_table(lut_plot)

            # 8) JSON 텍스트로 재조립 (TV write용)
            vac_json_optimized = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=new_lut_4096)

            # 9) 로딩 GIF 정지/완료 아이콘
            self._step_done(2)

            return vac_json_optimized, new_lut_4096

        except Exception as e:
            logging.exception("[PredictOpt] failed")
            # 로딩 애니 정리
            try:
                self._step_done(2)
            except Exception:
                pass
            return None, None

    def _predict_Y0W_from_models(self, lut256_dict, *, panel_text, frame_rate, model_year):
        """
        저장된 hybrid_*_model.pkl 3개로 'W' 패턴 256 포인트의 (Gamma, Cx, Cy) 예측 벡터를 생성
        """

        X_rows = [ self._build_runtime_feature_row_W(
                        lut256_dict, g,
                        panel_text=panel_text,
                        frame_rate=frame_rate,
                        model_year_2digit=float(model_year)  # 두 자리 숫자 가정
                ) for g in range(256) ]
        X = np.vstack(X_rows).astype(np.float32)

        def _pred_one(payload):
            lin = payload["linear_model"]; rf = payload["rf_residual"]
            tgt = payload["target_scaler"]; y_mean = float(tgt["mean"]); y_std = float(tgt["std"])
            base_s  = lin.predict(X).astype(np.float32)
            resid_s = rf.predict(X).astype(np.float32)
            y = (base_s + resid_s) * y_std + y_mean
            return y.astype(np.float32)

        yG  = _pred_one(self.models_Y0_bundle["Gamma"])
        yCx = _pred_one(self.models_Y0_bundle["Cx"])
        yCy = _pred_one(self.models_Y0_bundle["Cy"])

        # Gamma 0/255는 NaN
        yG[0] = np.nan; yG[255] = np.nan
        return {"Gamma": yG, "Cx": yCx, "Cy": yCy}


      
