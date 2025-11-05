def _after_off(store_off):
    self._off_store = store_off
    lv_off = np.zeros(256, dtype=np.float64)
    for g in range(256):
        tup = store_off['gamma']['main']['white'].get(g, None)
        lv_off[g] = float(tup[0]) if tup else np.nan

    # 기존: 전체 gamma series
    self._gamma_off_vec = self._compute_gamma_series(lv_off)

    # ★추가: 이후 ΔGamma 스펙 평가용으로 OFF 휘도 벡터 / max 저장
    self._lv_off_vec = lv_off.copy()
    try:
        self._lv_off_max = float(np.nanmax(lv_off[1:]))  # 0gray는 빼고 최대값
    except (ValueError, TypeError):
        self._lv_off_max = float('nan')

    self._step_done(1)
    ...