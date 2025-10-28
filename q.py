def _normalized_luminance(lv_vec):
    """
    lv_vec: (256,) 절대 휘도 [cd/m2]
    return: (256,) 0~1 정규화된 휘도
            Ynorm[g] = (Lv[g] - Lv[0]) / (max(Lv[1:]-Lv[0]))
    감마 계산과 동일한 노말라이제이션 방식 유지
    """
    lv_arr = np.asarray(lv_vec, dtype=np.float64)
    y0 = lv_arr[0]
    denom = np.nanmax(lv_arr[1:] - y0)
    if not np.isfinite(denom) or denom <= 0:
        return np.full(256, np.nan, dtype=np.float64)
    return (lv_arr - y0) / denom

def _block_slopes(lv_vec, g_start=88, g_stop=232, step=8):
    """
    lv_vec: (256,) 절대 휘도
    g_start..g_stop: 마지막 블록은 [224,232]까지 포함되도록 설정
    step: 8gray 폭

    return:
      mids  : (n_blocks,) 각 블록 중간 gray (예: 92,100,...,228)
      slopes: (n_blocks,) 각 블록의 slope
              slope = abs( Ynorm[g1] - Ynorm[g0] ) / ((g1-g0)/255)
              g0 = block start, g1 = block end (= g0+step)
    """
    Ynorm = _normalized_luminance(lv_vec)  # (256,)
    mids   = []
    slopes = []
    for g0 in range(g_start, g_stop, step):
        g1 = g0 + step
        if g1 >= len(Ynorm):
            break

        y0 = Ynorm[g0]
        y1 = Ynorm[g1]

        # 분모 = gray step을 0~1로 환산한 Δgray_norm
        d_gray_norm = (g1 - g0) / 255.0

        if np.isfinite(y0) and np.isfinite(y1) and d_gray_norm > 0:
            slope = abs(y1 - y0) / d_gray_norm
        else:
            slope = np.nan

        mids.append(g0 + (g1 - g0)/2.0)  # 예: 88~96 -> 92.0
        slopes.append(slope)

    return np.asarray(mids, dtype=np.float64), np.asarray(slopes, dtype=np.float64)