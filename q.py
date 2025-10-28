def _finalize_session(self):
    s = self._sess
    profile: SessionProfile = s['profile']
    table_main = self.ui.vac_table_opt_mes_results_main
    cols = profile.table_cols
    thr_gamma = 0.05

    # =========================
    # 1) main 감마 컬럼 채우기
    # =========================
    lv_series_main = np.zeros(256, dtype=np.float64)
    for g in range(256):
        tup = s['store']['gamma']['main']['white'].get(g, None)
        lv_series_main[g] = float(tup[0]) if tup else np.nan

    gamma_vec = self._compute_gamma_series(lv_series_main)
    for g in range(256):
        if np.isfinite(gamma_vec[g]):
            self._set_item(table_main, g, cols['gamma'], f"{gamma_vec[g]:.6f}")

    # =========================
    # 2) ΔGamma (ON세션일 때만)
    # =========================
    if profile.ref_store is not None and 'd_gamma' in cols:
        ref_lv_main = np.zeros(256, dtype=np.float64)
        for g in range(256):
            tup = profile.ref_store['gamma']['main']['white'].get(g, None)
            ref_lv_main[g] = float(tup[0]) if tup else np.nan
        ref_gamma = self._compute_gamma_series(ref_lv_main)
        dG = gamma_vec - ref_gamma
        for g in range(256):
            if np.isfinite(dG[g]):
                self._set_item_with_spec(
                    table_main, g, cols['d_gamma'], f"{dG[g]:.6f}",
                    is_spec_ok=(abs(dG[g]) <= thr_gamma)
                )

    # =================================================================
    # 3) [ADD: slope 계산 후 sub 테이블 업데이트 - 측정 종료 후 한 번에]
    # =================================================================
    # 요구사항:
    # - sub 측정 white의 lv로 normalized 휘도 계산
    # - 88gray부터 8 gray step씩 (88→96, 96→104, ... 224→232)
    # - slope = abs( Ynorm[g+8] - Ynorm[g] ) / ((8)/255)
    # - slope는 row=g 에 기록
    # - VAC OFF 세션이면 sub 테이블의 4번째 열(0-based index 3)
    #   VAC ON / CORR 세션이면 sub 테이블의 8번째 열(0-based index 7)

    table_sub = self.ui.vac_table_opt_mes_results_sub

    # 3-1) sub white lv 배열 뽑기
    lv_series_sub = np.full(256, np.nan, dtype=np.float64)
    for g in range(256):
        tup_sub = s['store']['gamma']['sub']['white'].get(g, None)
        if tup_sub:
            lv_series_sub[g] = float(tup_sub[0])

    # 3-2) 정규화된 휘도 Ynorm[g] = (Lv[g]-Lv[0]) / max(Lv[1:]-Lv[0])
    def _norm_lv(lv_arr):
        lv0 = lv_arr[0]
        denom = np.nanmax(lv_arr[1:] - lv0)
        if not np.isfinite(denom) or denom <= 0:
            return np.full_like(lv_arr, np.nan, dtype=np.float64)
        return (lv_arr - lv0) / denom

    Ynorm_sub = _norm_lv(lv_series_sub)

    # 3-3) 어느 열에 쓰는지 결정
    is_off_session = profile.legend_text.startswith('VAC OFF')
    slope_col_idx = 3 if is_off_session else 7  # 4번째 or 8번째 열

    # 3-4) 각 8gray 블록 slope 계산해서 테이블에 기록
    # 블록 시작 gray: 88,96,104,...,224
    for g0 in range(88, 225, 8):
        g1 = g0 + 8
        if g1 >= 256:
            break

        y0 = Ynorm_sub[g0]
        y1 = Ynorm_sub[g1]
        d_gray_norm = (g1 - g0) / 255.0  # 8/255

        if np.isfinite(y0) and np.isfinite(y1) and d_gray_norm > 0:
            slope_val = abs(y1 - y0) / d_gray_norm
            txt = f"{slope_val:.6f}"
        else:
            txt = ""

        # row = g0 에 기록
        self._set_item(table_sub, g0, slope_col_idx, txt)

    # 끝났으면 on_done 콜백 실행
    if callable(s['on_done']):
        try:
            s['on_done'](s['store'])
        except Exception as e:
            logging.exception(e)