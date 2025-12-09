def _ensure_vac_ml_components(self):
    """
    hybrid_Cx_model.pkl / hybrid_Cy_model.pkl / hybrid_Gamma_model.pkl 로딩 + shape 정보 로그 출력
    """
    # 이미 로딩되어 있으면 패스
    if hasattr(self, "_cx_payload") and self._cx_payload is not None:
        return

    logging.info("[ML] VAC 예측 모델 payload 로딩 시작")

    Cx_model_path = cf.get_normalized_path(__file__, '.', 'models', 'hybrid_Cx_model.pkl')
    Cy_model_path = cf.get_normalized_path(__file__, '.', 'models', 'hybrid_Cy_model.pkl')
    Gamma_model_path = cf.get_normalized_path(__file__, '.', 'models', 'hybrid_Gamma_model.pkl')

    try:
        self._cx_payload = joblib.load(Cx_model_path)
        self._cy_payload = joblib.load(Cy_model_path)
        self._gamma_payload = joblib.load(Gamma_model_path)
    except Exception:
        logging.exception("[ML] VAC 예측 모델 payload 로딩 실패")
        raise

    # 편의상 바로 꺼내서 멤버로도 들고 있게 해도 좋습니다.
    self._cx_linear  = self._cx_payload["linear_model"]
    self._cx_rf      = self._cx_payload["rf_residual"]
    self._cx_scaler  = self._cx_payload["target_scaler"]
    self._cx_schema  = self._cx_payload.get("feature_schema")

    self._cy_linear  = self._cy_payload["linear_model"]
    self._cy_rf      = self._cy_payload["rf_residual"]
    self._cy_scaler  = self._cy_payload["target_scaler"]
    self._cy_schema  = self._cy_payload.get("feature_schema")

    self._gamma_linear = self._gamma_payload["linear_model"]
    self._gamma_rf     = self._gamma_payload["rf_residual"]
    self._gamma_scaler = self._gamma_payload["target_scaler"]
    self._gamma_schema = self._gamma_payload.get("feature_schema")

    # ------------------------------------------------------------
    # ⬇⬇⬇ payload 별 shape / 구조 로그
    # ------------------------------------------------------------
    def _log_payload_info(name, payload):
        linear = payload["linear_model"]      # Pipeline(StandardScaler + Ridge)
        rf     = payload["rf_residual"]      # RandomForestRegressor
        t_scal = payload.get("target_scaler")
        schema = payload.get("feature_schema")

        logging.info(f"[ML] ===== {name} payload info =====")

        # 1) Linear (Pipeline) 구조 & feature 수
        logging.info(f"[ML] {name}.linear_model type = {type(linear)}")
        if hasattr(linear, "steps"):
            logging.info(f"[ML] {name}.linear_model steps = {[s[0] for s in linear.steps]}")
        if hasattr(linear, "n_features_in_"):
            logging.info(f"[ML] {name}.linear_model.n_features_in_ = {linear.n_features_in_}")

        # Ridge 내부 coef shape도 참고
        try:
            ridge = linear.named_steps.get("ridge", None)
        except Exception:
            ridge = None
        if ridge is not None and hasattr(ridge, "coef_"):
            logging.info(f"[ML] {name}.ridge.coef_.shape = {np.shape(ridge.coef_)}")

        # 2) RF residual 정보
        logging.info(f"[ML] {name}.rf_residual type = {type(rf)}")
        if hasattr(rf, "n_features_in_"):
            logging.info(f"[ML] {name}.rf_residual.n_features_in_ = {rf.n_features_in_}")
        if hasattr(rf, "n_estimators"):
            logging.info(f"[ML] {name}.rf_residual.n_estimators = {rf.n_estimators}")

        # 3) 타깃 스케일러 정보 (y mean/std)
        if t_scal is not None:
            logging.info(
                f"[ML] {name}.target_scaler "
                f"(mean={t_scal.get('mean'):.6f}, std={t_scal.get('std'):.6f}, "
                f"standardized={t_scal.get('standardized')})"
            )

        # 4) feature_schema (Y0 모델에만 들어 있음)
        if schema is not None:
            desc  = schema.get("desc", "")
            chs   = schema.get("channels", [])
            add_g = schema.get("add_gray_norm", False)
            add_p = schema.get("add_pattern_onehot", False)
            logging.info(f"[ML] {name}.feature_schema.desc = {desc}")
            logging.info(f"[ML] {name}.feature_schema.channels = {chs}")
            logging.info(
                f"[ML] {name}.feature_schema.add_gray_norm={add_g}, "
                f"add_pattern_onehot={add_p}"
            )

    _log_payload_info("Cx",    self._cx_payload)
    _log_payload_info("Cy",    self._cy_payload)
    _log_payload_info("Gamma", self._gamma_payload)
    # ------------------------------------------------------------

    # (필요하다면 여기서 VACInputBuilder / VACOutputBuilder도 생성)
    self._vac_input_builder = VACInputBuilder()
    self._vac_output_builder = VACOutputBuilder()

    logging.info("[ML] VAC 예측 모델 payload 로딩 및 shape 로그 완료")