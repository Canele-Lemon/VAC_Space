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
        여기서 아래 에러가 떴어요


2025-10-23 12:48:36,976 - DEBUG - subpage_vacspace.py:2192 - [UI META] panel='INX', fr='60Hz'→60.0, model_year='Y26'→26.0
2025-10-23 12:48:36,978 - DEBUG - subpage_vacspace.py:2252 - [RUNTIME X from DB+UI] shape=(256, 18), dim=18
2025-10-23 12:48:36,979 - DEBUG - subpage_vacspace.py:2269 - [RUNTIME X from DB+UI] panel_onehot sum unique: [1.] (expect 0 or 1)
2025-10-23 12:48:36,979 - DEBUG - subpage_vacspace.py:2270 - [RUNTIME X from DB+UI] ctx: panel='INX', fr=60.0, my(2digit)=26.0
2025-10-23 12:48:36,980 - DEBUG - subpage_vacspace.py:2278 - [RUNTIME X from DB+UI] sample: idx=  0 | LUT6=[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] | tail10=[0.0000, 0.0000, 1.0000, 60.0000, 26.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000]
2025-10-23 12:48:36,980 - DEBUG - subpage_vacspace.py:2279 - [RUNTIME X from DB+UI] sample: idx=128 | LUT6=[0.1800, 0.6869, 0.2415, 0.6801, 0.1499, 0.6869] | tail10=[0.0000, 0.0000, 1.0000, 60.0000, 26.0000, 0.5020, 1.0000, 0.0000, 0.0000, 0.0000]
2025-10-23 12:48:36,980 - DEBUG - subpage_vacspace.py:2280 - [RUNTIME X from DB+UI] sample: idx=255 | LUT6=[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000] | tail10=[0.0000, 0.0000, 1.0000, 60.0000, 26.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000]
2025-10-23 12:48:36,980 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=246 | gray_norm=0.9647 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9647058844566345, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,981 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=247 | gray_norm=0.9686 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9686274528503418, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,981 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=248 | gray_norm=0.9725 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9725490212440491, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,981 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=249 | gray_norm=0.9765 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9764705896377563, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,981 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=250 | gray_norm=0.9804 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9803921580314636, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,982 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=251 | gray_norm=0.9843 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9843137264251709, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,982 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=252 | gray_norm=0.9882 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9882352948188782, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,982 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=253 | gray_norm=0.9922 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9921568632125854, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,982 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=254 | gray_norm=0.9961 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 0.9960784316062927, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,982 - DEBUG - subpage_vacspace.py:2286 - [RUNTIME X from DB+UI] last10 idx=255 | gray_norm=1.0000 | tail10=(0.0, 0.0, 1.0, 60.0, 26.0, 1.0, 1.0, 0.0, 0.0, 0.0)
2025-10-23 12:48:36,982 - INFO - subpage_vacspace.py:937 - [PredictOpt] 예측 기반 1차 최적화 시작
2025-10-23 12:48:36,987 - DEBUG - subpage_vacspace.py:2192 - [UI META] panel='INX', fr='60Hz'→60.0, model_year='Y26'→26.0
2025-10-23 12:48:36,989 - ERROR - subpage_vacspace.py:2129 - [PredictOpt] failed
Traceback (most recent call last):
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 2073, in _predictive_first_optimize
    y_pred = self._predict_Y0W_from_models(lut256_iter,
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1987, in _predict_Y0W_from_models
    X_rows = [ self._build_runtime_feature_row_W(lut256_dict, g,
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1987, in <listcomp>
    X_rows = [ self._build_runtime_feature_row_W(lut256_dict, g,
TypeError: Widget_vacspace._build_runtime_feature_row_W() got an unexpected keyword argument 'model_year'
2025-10-23 12:48:36,992 - WARNING - subpage_vacspace.py:942 - [PredictOpt] 실패 → 원본 DB LUT로 진행


