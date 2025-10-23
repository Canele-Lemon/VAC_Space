def _apply_vac_from_db_and_measure_on(self):
    ...
    vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
    if vac_data is None:
        logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
        return

    # ✅ 0) OFF 끝났고, 여기서 1차 예측 최적화 먼저 수행
    logging.info("[PredictOpt] 예측 기반 1차 최적화 시작")
    vac_data_to_write, _lut4096_dict = self._predictive_first_optimize(
        vac_data, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3
    )
    if vac_data_to_write is None:
        logging.warning("[PredictOpt] 실패 → 원본 DB LUT로 진행")
        vac_data_to_write = vac_data
    else:
        logging.info("[PredictOpt] 1차 최적화 LUT 생성 완료 → UI 업데이트 반영됨")

    # TV 쓰기 완료 시 콜백
    def _after_write(ok, msg):
        ...
    ...
    logging.info("[LUT LOADING] (예측 최적화 LUT) TV Writing 시작")
    self._write_vac_to_tv(vac_data_to_write, on_finished=_after_write)