if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
    ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
    if ref_main is not None and np.isfinite(cx_m) and np.isfinite(cy_m):
        _, cx_r, cy_r = ref_main
        d_cx = cx_m - cx_r
        d_cy = cy_m - cy_r

        # ★ 스펙 판정은 반올림 기준
        cx_ok = round(abs(d_cx), 4) <= 0.003   # ΔCx
        cy_ok = round(abs(d_cy), 4) <= 0.003   # ΔCy

        if gray in (0, 1, 254, 255):
            # edge gray → 색깔은 건드리지 않고 값만 채워 넣음
            self._set_item(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}")
            self._set_item(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}")
        else:
            # 나머지 gray만 빨강/파랑 표시
            self._set_item_with_spec(
                table_inst1, gray, cols['d_cx'],
                f"{d_cx:.6f}", is_spec_ok=cx_ok
            )
            self._set_item_with_spec(
                table_inst1, gray, cols['d_cy'],
                f"{d_cy:.6f}", is_spec_ok=cy_ok
            )