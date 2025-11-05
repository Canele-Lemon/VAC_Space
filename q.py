def _after_off(store_off):
    self._off_store = store_off

    # OFF 전체 Lv는 감마 벡터 계산용으로만 사용
    lv_off = np.zeros(256, dtype=np.float64)
    for g in range(256):
        tup = store_off['gamma']['main']['white'].get(g, None)
        lv_off[g] = float(tup[0]) if tup else np.nan

    # 🔹 OFF 기준 감마 시리즈만 캐싱 (타깃용)
    self._gamma_off_vec = self._compute_gamma_series(lv_off)

    self._step_done(1)
    logging.info("[Measurement] VAC OFF 상태 측정 완료")

    logging.info("[TV Control] VAC ON 전환 시작")
    if not self._set_vac_active(True):
        logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
        return
    logging.info("[TV Control] VAC ON 전환 성공")
    # ...