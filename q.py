G_off = self._compute_gamma_series(lv_off)
G_on  = self._compute_gamma_series(lv_on)

# 부호 유지해 두고, abs는 밑에서 씁니다.
dG  = G_on - G_off        # (256,)
dCx = cx_on - cx_off
dCy = cy_on - cy_off

def _pass_total_chroma(d_arr, thr):
    # 유효 값 + edge gray(0,1,254,255) 제외
    mask = np.isfinite(d_arr)
    for g in (0, 1, 254, 255):
        if 0 <= g < len(mask):
            mask[g] = False

    vals = d_arr[mask]
    tot = int(np.sum(mask))
    if tot <= 0:
        return 0, 0

    # ★ 소수점 4째 자리에서 반올림
    rounded = np.round(np.abs(vals), 4)
    thr_r = round(float(thr), 4)
    ok = int(np.sum(rounded <= thr_r))
    return ok, tot

def _pass_total_gamma(d_arr, thr):
    mask = np.isfinite(d_arr)
    for g in (0, 1, 254, 255):
        if 0 <= g < len(mask):
            mask[g] = False

    vals = d_arr[mask]
    tot = int(np.sum(mask))
    if tot <= 0:
        return 0, 0

    # ★ 소수점 3째 자리에서 반올림
    rounded = np.round(np.abs(vals), 3)
    thr_r = round(float(thr), 3)
    ok = int(np.sum(rounded <= thr_r))
    return ok, tot

ok_cx, tot_cx = _pass_total_chroma(dCx, thr_c)
ok_cy, tot_cy = _pass_total_chroma(dCy, thr_c)
ok_g , tot_g  = _pass_total_gamma(dG , thr_gamma)