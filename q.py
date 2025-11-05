def _update_last_on_lv_norm(self, on_store):
    """
    마지막 전체 ON 측정 결과(on_store)에서
    Lv[0], max(Lv[1:]-Lv[0])를 구해 fine 보정용으로 저장.
    """
    lv_on = np.full(256, np.nan, np.float64)
    for g in range(256):
        tup = on_store['gamma']['main']['white'].get(g, None)
        if tup:
            lv_on[g] = float(tup[0])

    lv0 = lv_on[0]
    with np.errstate(invalid='ignore'):
        denom = np.nanmax(lv_on[1:] - lv0) if np.isfinite(lv0) else np.nan

    if (not np.isfinite(denom)) or denom <= 0:
        logging.warning("[FineNorm] invalid ON Lv norm (denom<=0) → fine gamma disabled")
        self._fine_lv0_on = float("nan")
        self._fine_denom_on = float("nan")
    else:
        self._fine_lv0_on = float(lv0)
        self._fine_denom_on = float(denom)
        logging.info(f"[FineNorm] updated from last ON: Lv0={lv0:.3f}, denom={denom:.3f}")