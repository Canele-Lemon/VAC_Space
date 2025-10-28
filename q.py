def _compute_block_slope_for_gray(self, store_role_dict, g0, step=8):
    """
    store_role_dict: store['gamma']['sub']['white'] 같은 dict.
                     key = gray(int), value = (lv, cx, cy)
    g0: 시작 gray
    step: 구간 폭 (8)

    return:
        slope (float) or np.nan
        정의: |Ynorm[g1]-Ynorm[g0]| / ((g1-g0)/255)
              Ynorm[g] = (Lv[g]-Lv[0]) / max(Lv[1:]-Lv[0])
    """

    g1 = g0 + step
    # 필수 샘플이 없으면 slope 못 구함
    if g0 not in store_role_dict or g1 not in store_role_dict:
        return np.nan

    # lv 배열(0~255)을 한 번 만들어서 정규화해야 해요.
    # 여기서는 sub 전체 white 데이터를 기반으로 한다고 하셨으므로
    # store_role_dict 전체에서 lv를 뽑아와 256짜리 벡터를 만들자.
    lv_arr = np.full(256, np.nan, dtype=np.float64)
    for gg, tup in store_role_dict.items():
        if tup is None:
            continue
        lv_arr[int(gg)] = float(tup[0])

    lv0 = lv_arr[0]
    denom = np.nanmax(lv_arr[1:] - lv0)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan

    # 정규화된 휘도
    y0 = (lv_arr[g0] - lv0) / denom
    y1 = (lv_arr[g1] - lv0) / denom

    if not (np.isfinite(y0) and np.isfinite(y1)):
        return np.nan

    d_gray_norm = (g1 - g0) / 255.0
    if d_gray_norm <= 0:
        return np.nan

    slope = abs(y1 - y0) / d_gray_norm
    return float(slope)