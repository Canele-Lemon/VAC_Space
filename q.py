def _predict_and_correct_vac(self, base_vac_dict):
    """
    Base VAC LUT에 대해:
      1) ML 모델로 ΔCx, ΔCy, ΔGamma 예측
      2) 예측 결과를 이용해 LUT를 한 번 보정
    반환: 보정된 vac_dict
    """
    # ------------------------------------------------------
    # 1) feature vector 구성
    # ------------------------------------------------------
    # off_store: VAC OFF 측정 결과 (이미 _run_off_baseline_then_on()에서 채워진다고 가정)
    off_store = getattr(self, "_off_store", None)
    if off_store is None:
        logging.warning("[ML] off_store가 없습니다. Base VAC만 그대로 사용합니다.")
        return base_vac_dict

    # 패널/프레임레이트/연도 등 메타 정보
    panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
    fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
    try:
        fr = int(fr_text)
    except ValueError:
        fr = 60  # fallback

    prod_year_text = getattr(self.ui, "vac_cmb_ProdYear", None)
    if prod_year_text is not None:
        prod_year = self.ui.vac_cmb_ProdYear.currentText().strip()
    else:
        prod_year = "24"  # 필요시 수정

    # VACInputBuilder 쪽에 맞는 인자 이름으로 넣어주세요.
    # 예: build_input(panel, fr, prod_year, vac_dict, off_store, ...)
    X = self._vac_input_builder.build_input(
        vac_dict=base_vac_dict,
        off_store=off_store,
        panel=panel,
        frame_rate=fr,
        prod_year=prod_year,
    )
    # shape: (1, n_features) 또는 (n_samples, n_features) 여야 함

    # ------------------------------------------------------
    # 2) Cx / Cy / Gamma 예측
    # ------------------------------------------------------
    try:
        y_cx_pred = self._cx_model.predict(X)
        y_cy_pred = self._cy_model.predict(X)
        y_gamma_pred = self._gamma_model.predict(X)
    except Exception:
        logging.exception("[ML] VAC 예측 중 오류 발생 - Base VAC만 사용합니다.")
        return base_vac_dict

    # ------------------------------------------------------
    # 3) 예측값을 기반으로 LUT 보정
    # ------------------------------------------------------
    # VACOutputBuilder 에 맞게 포맷 조립
    # (여기 부분은 실제 구현에 맞게 수정 필요)
    pred_dict = self._vac_output_builder.build_pred_dict(
        y_cx=y_cx_pred,
        y_cy=y_cy_pred,
        y_gamma=y_gamma_pred,
    )

    # 예: output_builder.apply_correction(base_vac_dict, pred_dict, spec, ...)
    corrected_vac_dict = self._vac_output_builder.apply_correction(
        base_vac_dict=base_vac_dict,
        pred_metrics=pred_dict,
        thr_gamma=0.05,
        thr_c=0.003,
    )

    # 0~4095 clip, monotonic 보장 등 후처리가 있다면 여기에서
    corrected_vac_dict = self._vac_output_builder.postprocess_lut(corrected_vac_dict)

    logging.info("[ML] Base VAC에 대한 1차 예측/보정 완료")
    return corrected_vac_dict