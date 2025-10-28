def _consume_gamma_pair(self, pattern, gray, results):
    """
    results: {
        'main': (x, y, lv, cct, duv)  또는  None,
        'sub' : (x, y, lv, cct, duv)  또는  None
    }
    """
    s = self._sess
    store = s['store']
    profile: SessionProfile = s['profile']

    state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

    # 1) raw 측정값을 store에 적재 + gamma_chart 업데이트 (main/sub 공통)
    for role in ('main', 'sub'):
        res = results.get(role, None)
        if res is None:
            store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
            continue

        x, y, lv, cct, duv = res
        store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

        # 감마 차트(휘도 vs gray) 갱신은 유지
        self.vac_optimization_gamma_chart.add_point(
            state=state,
            role=role,
            pattern=pattern,
            gray=int(gray),
            luminance=float(lv)
        )

    # 2) white일 때 테이블 갱신
    if pattern == 'white':
        cols = profile.table_cols

        # ----- main 테이블 (기존 동작 유지) -----
        table_main = self.ui.vac_table_opt_mes_results_main
        lv_m, cx_m, cy_m = store['gamma']['main']['white'].get(gray, (np.nan, np.nan, np.nan))

        # Lv / Cx / Cy 갱신
        self._set_item(table_main, gray, cols['lv'],
                       f"{lv_m:.6f}" if np.isfinite(lv_m) else "")
        self._set_item(table_main, gray, cols['cx'],
                       f"{cx_m:.6f}" if np.isfinite(cx_m) else "")
        self._set_item(table_main, gray, cols['cy'],
                       f"{cy_m:.6f}" if np.isfinite(cy_m) else "")

        # ΔCx/ΔCy (main 쪽은 계속 표시)
        if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
            ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
            if ref_main is not None:
                _, cx_r, cy_r = ref_main
                if np.isfinite(cx_m):
                    d_cx = cx_m - cx_r
                    self._set_item_with_spec(
                        table_main, gray, cols['d_cx'], f"{d_cx:.6f}",
                        is_spec_ok=(abs(d_cx) <= 0.003)
                    )
                if np.isfinite(cy_m):
                    d_cy = cy_m - cy_r
                    self._set_item_with_spec(
                        table_main, gray, cols['d_cy'], f"{d_cy:.6f}",
                        is_spec_ok=(abs(d_cy) <= 0.003)
                    )

        # ----- sub 테이블 (요구사항대로 변경) -----
        table_sub = self.ui.vac_table_opt_mes_results_sub
        sub_white_dict = store['gamma']['sub']['white']
        # lv_s, cx_s, cy_s = sub_white_dict.get(gray, (np.nan, np.nan, np.nan))

        # ❌ 더 이상 sub 테이블에 lv/cx/cy/dCx/dCy/gamma 안 찍는다.
        #    -> 지우고 대신 '휘도 기울기'만 넣는다.

        # 우리가 테이블에서 이 "휘도 기울기"를 넣고 싶은 열 index를 정해야 해요.
        # 가령 profile.table_cols['gamma'] 열을 재활용해도 되고,
        # 새로 profile.table_cols['slope'] 를 하나 예약해도 됩니다.
        # 아래에선 'gamma' 자리를 재활용한다고 가정할게요.
        slope_col = cols.get('gamma', None)  # sub 테이블에서 기울기 표시용 열

        if slope_col is not None:
            slope_val = self._compute_block_slope_for_gray(
                sub_white_dict,
                g0=int(gray),
                step=8
            )

            # slope는 88,96,...224행에서만 유효해야 정상입니다.
            # g0+8이 255 넘어가거나 아직 두 점이 없으면 NaN이 나옵니다.
            txt = f"{slope_val:.6f}" if np.isfinite(slope_val) else ""
            # 일반 set_item이면 충분 (spec 판정 없음)
            self._set_item(table_sub, gray, slope_col, txt)

        # ✅ Cx/Cy 델타, 감마 등은 sub 테이블에 더 이상 쓰지 않음
        # (아무 것도 안 함)