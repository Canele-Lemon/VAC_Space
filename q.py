vac_table_opt_mes_results_sub여기에 값이 업데이트 되는 방식도 vac_table_opt_mes_results_main와는 다르게 가려고 해요.

현재는 두 표 모두 측정이 끝나면 감마가 계산되잖아요

vac_table_opt_mes_results_sub는 감마가 계산되는 것이 아닌, 현재 감마가 계산되는 열에 앞서 말씀드린 휘도 기울기가 계산되어야 합니다. 또 Cx, Cy 감마 델타는 계산할 필요가 없습니다. 단지 휘도 기울기만 업데이트되면 돼요

아마 코드에서는 이 부분인거 같습니다. (아닐수도 있어요)

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

        # 현재 세션이 OFF 레퍼런스인지, ON/보정 런인지 상태 문자열 결정
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        # 두 역할을 results 키로 직접 순회 (측정기 객체 비교 X)
        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                # 측정 실패/결측인 경우
                store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
                continue

            x, y, lv, cct, duv = res
            # 스토어 업데이트 (white 테이블/감마 계산 등에 사용)
            store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

            # ▶▶ 차트 업데이트 (간소화 API)
            # GammaChartVAC: add_point(state, role, pattern, gray, luminance)
            self.vac_optimization_gamma_chart.add_point(
                state=state,
                role=role,               # 'main' 또는 'sub'
                pattern=pattern,         # 'white'|'red'|'green'|'blue'
                gray=int(gray),
                luminance=float(lv)
            )

        # (아래 white/main 테이블 채우는 로직은 기존 그대로 유지)
        if pattern == 'white':
            # main 테이블
            lv_m, cx_m, cy_m = store['gamma']['main']['white'].get(gray, (np.nan, np.nan, np.nan))
            table_inst1 = self.ui.vac_table_opt_mes_results_main
            cols = profile.table_cols
            self._set_item(table_inst1, gray, cols['lv'], f"{lv_m:.6f}" if np.isfinite(lv_m) else "")
            self._set_item(table_inst1, gray, cols['cx'], f"{cx_m:.6f}" if np.isfinite(cx_m) else "")
            self._set_item(table_inst1, gray, cols['cy'], f"{cy_m:.6f}" if np.isfinite(cy_m) else "")

            # sub 테이블
            lv_s, cx_s, cy_s = store['gamma']['sub']['white'].get(gray, (np.nan, np.nan, np.nan))
            table_inst2 = self.ui.vac_table_opt_mes_results_sub
            self._set_item(table_inst2, gray, cols['lv'], f"{lv_s:.6f}" if np.isfinite(lv_s) else "")
            self._set_item(table_inst2, gray, cols['cx'], f"{cx_s:.6f}" if np.isfinite(cx_s) else "")
            self._set_item(table_inst2, gray, cols['cy'], f"{cy_s:.6f}" if np.isfinite(cy_s) else "")

            # ΔCx/ΔCy (ON 세션에서만; ref_store가 있을 때)                    
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

