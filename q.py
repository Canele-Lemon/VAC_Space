lv_off = np.zeros(256, dtype=np.float64)
for g in range(256):
    tup = store_off['gamma']['main']['white'].get(g, None)
    lv_off[g] = float(tup[0]) if tup else np.nan
self._off_lv_vec = lv_off
self._gamma_off_vec = self._compute_gamma_series(lv_off)  # 참고용(최종평가에도 사용)
self._off_lv0 = float(lv_off[0])
self._off_denom = float(np.nanmax(lv_off[1:] - self._off_lv0))