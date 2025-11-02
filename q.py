if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
    ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
    if ref_main is not None and np.isfinite(cx_m) and np.isfinite(cy_m):
        _, cx_r, cy_r = ref_main
        d_cx = cx_m - cx_r
        d_cy = cy_m - cy_r
        self._set_item_with_spec(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}", is_spec_ok=(abs(d_cx) <= 0.003))
        self._set_item_with_spec(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}", is_spec_ok=(abs(d_cy) <= 0.003))