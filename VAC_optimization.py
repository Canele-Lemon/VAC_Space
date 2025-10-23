    def _predictive_first_optimize(self, vac_data_json, *, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3):
        """
        DB에서 가져온 VAC JSON을 예측모델+자코비안으로 미리 n회 보정.
        - 감마 정확도 낮음: wG(기본 0.4)로 영향 축소
        - return: (optimized_vac_json_str, lut_dict_4096)  혹은 (None, None) 실패 시
        """
        try:
            # 1) json→dict
            vac_dict = json.loads(vac_data_json)

            # 2) 4096→256
            lut256 = {
                "R_Low":  self._down4096_to_256_float(vac_dict["RchannelLow"]),
                "R_High": self._down4096_to_256_float(vac_dict["RchannelHigh"]),
                "G_Low":  self._down4096_to_256_float(vac_dict["GchannelLow"]),
                "G_High": self._down4096_to_256_float(vac_dict["GchannelHigh"]),
                "B_Low":  self._down4096_to_256_float(vac_dict["BchannelLow"]),
                "B_High": self._down4096_to_256_float(vac_dict["BchannelHigh"]),
            }

            # 3) 자코비안 준비 확인
            if not hasattr(self, "A_Gamma"):  # start에서 이미 _load_jacobian_artifacts 호출됨
                logging.error("[PredictOpt] Jacobian not prepared.")
                return None, None

            # 4) UI 메타
            panel, fr, model_year = self._get_ui_meta()

            # 5) 반복 보정
            K = len(self._jac_artifacts["knots"])
            Phi = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)

            high_R = lut256["R_High"].copy()
            high_G = lut256["G_High"].copy()
            high_B = lut256["B_High"].copy()

            for it in range(1, n_iters+1):
                # (a) 예측 ON (W)
                lut256_iter = {
                    "R_Low": lut256["R_Low"], "G_Low": lut256["G_Low"], "B_Low": lut256["B_Low"],
                    "R_High": high_R, "G_High": high_G, "B_High": high_B
                }
                y_pred = self._predict_Y0W_from_models(lut256_iter,
                            panel_text=panel, frame_rate=fr, model_year=model_year)

                # (b) Δ타깃 (예측 ON vs OFF)
                d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)

                # (c) 결합 선형계
                A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
                b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)
                mask = np.isfinite(b_cat)
                A_use = A_cat[mask,:]; b_use = b_cat[mask]

                ATA = A_use.T @ A_use
                rhs = A_use.T @ b_use
                ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
                delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

                # (d) 채널별 Δcurve = Phi @ Δh
                dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
                corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

                # (e) High 갱신 + 보정
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

            # 7) UI 바로 업데이트 (차트+테이블)
            lut_plot = {
                "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
                "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
                "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
            }
            self._update_lut_chart_and_table(lut_plot)

            # 8) JSON 텍스트로 재조립 (TV write용)
            vac_json_optimized = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=new_lut_4096)

            # 9) 로딩 GIF 정지/완료 아이콘
            self.stop_loading_animation(self.label_processing_step_2, self.movie_processing_step_2)
            self.ui.vac_label_pixmap_step_2.setPixmap(self.process_complete_pixmap)

            return vac_json_optimized, new_lut_4096

    def _predict_Y0W_from_models(self, lut256_dict, *, panel_text, frame_rate, model_year):
        """
        저장된 hybrid_*_model.pkl 3개로 'W' 패턴 256 포인트의 (Gamma, Cx, Cy) 예측 벡터를 생성
        """
        # 256행 피처 매트릭스
        X_rows = [ self._build_runtime_feature_row_W(lut256_dict, g,
                        panel_text=panel_text, frame_rate=frame_rate, model_year=model_year) for g in range(256) ]
        X = np.vstack(X_rows).astype(np.float32)

        def _pred_one(payload):
            lin = payload["linear_model"]; rf = payload["rf_residual"]
            tgt = payload["target_scaler"]; y_mean = float(tgt["mean"]); y_std = float(tgt["std"])
            base_s = lin.predict(X)
            resid_s = rf.predict(X).astype(np.float32)
            y = (base_s + resid_s) * y_std + y_mean
            return y.astype(np.float32)

        yG = _pred_one(self.models_Y0_bundle["Gamma"])
        yCx= _pred_one(self.models_Y0_bundle["Cx"])
        yCy= _pred_one(self.models_Y0_bundle["Cy"])
        # Gamma의 0,255는 신뢰구간 밖 → NaN 취급
        yG[0] = np.nan; yG[255] = np.nan
        return {"Gamma": yG, "Cx": yCx, "Cy": yCy}


Traceback (most recent call last):
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 2073, in _predictive_first_optimize
    y_pred = self._predict_Y0W_from_models(lut256_iter,
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1987, in _predict_Y0W_from_models
    X_rows = [ self._build_runtime_feature_row_W(lut256_dict, g,
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1987, in <listcomp>
    X_rows = [ self._build_runtime_feature_row_W(lut256_dict, g,
TypeError: Widget_vacspace._build_runtime_feature_row_W() got an unexpected keyword argument 'model_year'
2025-10-23 13:13:59,427 - WARNING - subpage_vacspace.py:942 - [PredictOpt] 실패 → 원본 DB LUT로 진행


여기서 계속 아래 에러가 발생하는데 뭘 고치면 되나요
