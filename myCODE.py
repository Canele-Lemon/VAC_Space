# (1) ΔCx/ΔCy: _consume_gamma_pair 안의 white 처리 부분
if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
    ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
    if ref_main is not None:
        _, cx_r, cy_r = ref_main
        if np.isfinite(cx_m):
            d_cx = cx_m - cx_r
            self._set_item_with_spec(
                table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}",
                is_spec_ok=(abs(d_cx) <= 0.003)  # thr_c
            )
        if np.isfinite(cy_m):
            d_cy = cy_m - cy_r
            self._set_item_with_spec(
                table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}",
                is_spec_ok=(abs(d_cy) <= 0.003)  # thr_c
            )
    # sub도 동일하게 적용
    ref_sub = profile.ref_store['gamma']['sub']['white'].get(gray, None)
    if ref_sub is not None:
        _, cx_r_s, cy_r_s = ref_sub
        if np.isfinite(cx_s):
            d_cx_s = cx_s - cx_r_s
            self._set_item_with_spec(
                table_inst2, gray, cols['d_cx'], f"{d_cx_s:.6f}",
                is_spec_ok=(abs(d_cx_s) <= 0.003)
            )
        if np.isfinite(cy_s):
            d_cy_s = cy_s - cy_r_s
            self._set_item_with_spec(
                table_inst2, gray, cols['d_cy'], f"{d_cy_s:.6f}",
                is_spec_ok=(abs(d_cy_s) <= 0.003)
            )
            
# (2) ΔGamma: _finalize_session 의 dG 갱신 부분
thr_gamma = 0.05
for g in range(256):
    if np.isfinite(dG[g]):
        self._set_item_with_spec(
            table, g, cols['d_gamma'], f"{dG[g]:.6f}",
            is_spec_ok=(abs(dG[g]) <= thr_gamma)
        )