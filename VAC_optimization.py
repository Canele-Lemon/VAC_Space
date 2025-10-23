def _predict_Y0W_from_models(self, lut256_dict, *, panel_text, frame_rate, model_year):
    """
    저장된 hybrid_*_model.pkl 3개로 'W' 패턴 256 포인트의 (Gamma, Cx, Cy) 예측 벡터를 생성
    """
    # ✅ LUT는 반드시 0..1 스케일로 맞춘다
    def _norm01(a): 
        return np.clip(np.asarray(a, np.float32) / 4095.0, 0.0, 1.0)
    lut256_norm = {
        "R_Low":  _norm01(lut256_dict["R_Low"]),
        "R_High": _norm01(lut256_dict["R_High"]),
        "G_Low":  _norm01(lut256_dict["G_Low"]),
        "G_High": _norm01(lut256_dict["G_High"]),
        "B_Low":  _norm01(lut256_dict["B_Low"]),
        "B_High": _norm01(lut256_dict["B_High"]),
    }

    # 256행 피처 매트릭스
    # ✅ _build_runtime_feature_row_W의 파라미터명은 model_year_2digit 입니다.
    X_rows = [ self._build_runtime_feature_row_W(
                    lut256_norm, g,
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
    
    
    
def _predictive_first_optimize(self, vac_data_json, *, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3):
    try:
        vac_dict = json.loads(vac_data_json)

        # ⬇️ 12bit(0..4095) 기준 다운샘플: 보정/적용용
        lut256_12 = {
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
        Phi = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)

        # 보정은 12bit에서 진행
        high_R_12 = lut256_12["R_High"].copy()
        high_G_12 = lut256_12["G_High"].copy()
        high_B_12 = lut256_12["B_High"].copy()

        for it in range(1, n_iters+1):
            # ✅ 예측용: 0..1 스케일로 변환해서 전달
            lut256_for_pred = {
                "R_Low":  lut256_12["R_Low"],
                "G_Low":  lut256_12["G_Low"],
                "B_Low":  lut256_12["B_Low"],
                "R_High": high_R_12,
                "G_High": high_G_12,
                "B_High": high_B_12,
            }
            y_pred = self._predict_Y0W_from_models(
                lut256_for_pred, panel_text=panel, frame_rate=fr, model_year=model_year
            )

            # (선택) 덤프
            self._debug_dump_predicted_Y0W(y_pred, tag=f"iter{it}_{panel}_fr{int(fr)}_my{int(model_year)%100:02d}", save_csv=True)

            # Δ타깃
            d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)

            # 결합 선형계
            A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
            b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)
            mask  = np.isfinite(b_cat)
            A_use = A_cat[mask,:]; b_use = b_cat[mask]

            ATA = A_use.T @ A_use
            rhs = A_use.T @ b_use
            ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
            delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

            # Δcurve(자코비안 출력은 12bit 스케일) → High(12bit)에 더함
            dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
            corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

            high_R_12 = np.clip(self._enforce_monotone(high_R_12 + corr_R), 0, 4095)
            high_G_12 = np.clip(self._enforce_monotone(high_G_12 + corr_G), 0, 4095)
            high_B_12 = np.clip(self._enforce_monotone(high_B_12 + corr_B), 0, 4095)

            logging.info(f"[PredictOpt] iter {it} done. (wG={wG}, wC={wC})")

        # 256→4096 업샘플 (Low는 원본 12bit 유지, High는 갱신된 12bit 사용)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"], dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"], dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"], dtype=np.float32),
            "RchannelHigh": self._up256_to_4096(high_R_12),
            "GchannelHigh": self._up256_to_4096(high_G_12),
            "BchannelHigh": self._up256_to_4096(high_B_12),
        }

        # UI 업데이트, JSON 재조립 등은 기존 그대로...