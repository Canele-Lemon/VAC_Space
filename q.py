def _normalize_lv_series(self, lv_vec_256, *, eps=0.0):
    """
    Lv(256) -> normalized Y(256)

    Y = (Lv - Lv0) / max(Lv[1:] - Lv0)

    - Lv0: gray 0 휘도
    - denom: (Lv[1:] - Lv0)의 최대값
    - denom이 0/NaN이면 전부 NaN 반환
    - eps>0이면 Y를 [eps, 1-eps]로 클리핑 (gamma 계산 안정화용)
    """
    lv = np.asarray(lv_vec_256, dtype=np.float64)
    y = np.full(256, np.nan, dtype=np.float64)

    if lv.size < 256:
        # 방어(필요하면 제거 가능)
        tmp = np.full(256, np.nan, dtype=np.float64)
        tmp[:lv.size] = lv
        lv = tmp

    lv0 = lv[0]
    denom = np.nanmax(lv[1:] - lv0)
    if not np.isfinite(denom) or denom <= 0:
        return y

    y = (lv - lv0) / denom

    if eps and eps > 0:
        y = np.clip(y, eps, 1.0 - eps)

    return y