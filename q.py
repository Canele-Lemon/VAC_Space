            # ΔCx/ΔCy (ON 세션에서만; ref_store가 있을 때)                    
            if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
                ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
                if ref_main is not None and np.isfinite(cx_m) and np.isfinite(cy_m):
                    _, cx_r, cy_r = ref_main
                    d_cx = cx_m - cx_r
                    d_cy = cy_m - cy_r

                    cx_ok = round(abs(d_cx), 4) <= 0.003
                    cy_ok = round(abs(d_cy), 4) <= 0.003

                    if gray in (0, 1, 254, 255):
                        self._set_item(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}")
                        self._set_item(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}")
                    else:
                        self._set_item_with_spec(
                            table_inst1, gray, cols['d_cx'],
                            f"{d_cx:.6f}", is_spec_ok=cx_ok
                        )
                        self._set_item_with_spec(
                            table_inst1, gray, cols['d_cy'],
                            f"{d_cy:.6f}", is_spec_ok=cy_ok
                        )

                    # ★ 여기서 ΔGamma도 간단히 추가
                    if 'd_gamma' in cols and hasattr(self, "_gamma_off_vec"):
                        # 1) 현재까지의 ON 휘도 벡터 구성
                        lv_on = np.zeros(256, dtype=np.float64)
                        for gg in range(256):
                            tup_on = store['gamma']['main']['white'].get(gg, None)
                            lv_on[gg] = float(tup_on[0]) if tup_on else np.nan

                        # 2) ON gamma 전체 계산
                        gamma_on_vec = self._compute_gamma_series(lv_on)

                        # 3) 현재 gray의 ΔGamma
                        g_off = self._gamma_off_vec[gray] if self._gamma_off_vec is not None else np.nan
                        g_on  = gamma_on_vec[gray]
                        if np.isfinite(g_off) and np.isfinite(g_on):
                            d_gamma = g_on - g_off

                            if gray in (0, 1, 254, 255):
                                # edge gray → 색깔 안 바꾸고 값만
                                self._set_item(
                                    table_inst1, gray, cols['d_gamma'],
                                    f"{d_gamma:.6f}"
                                )
                            else:
                                g_ok = round(abs(d_gamma), 3) <= 0.05
                                self._set_item_with_spec(
                                    table_inst1, gray, cols['d_gamma'],
                                    f"{d_gamma:.6f}", is_spec_ok=g_ok
                                )