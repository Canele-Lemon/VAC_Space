    def _consume_colorshift_pair(self, patch_name, results):
        """
        results: {
            'main': (x, y, lv, cct, duv)  또는  None,
            'sub' : (x, y, lv, cct, duv)  또는  None
        }
        """
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        # 현재 세션 상태 문자열
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        # 현재 패턴의 row index = cs_idx (op.colorshift_patterns 순서 그대로)
        row_idx = s['cs_idx']

        # 미리 테이블 핸들 가져오기
        tbl_cs_raw = self.ui.vac_table_opt_mes_results_colorshift

        # main/sub 측정 결과 기록하고 차트 갱신 (기존 코드 유지)
        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                store['colorshift'][role].append((np.nan, np.nan, np.nan, np.nan))
                continue

            x, y, lv, cct, duv_unused = res

            # xy -> u' v'
            u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))

            # store에 누적 (x,y,u',v')
            store['colorshift'][role].append((float(x), float(y), float(u_p), float(v_p)))

            # 차트 갱신 (기존)
            self.vac_optimization_cie1976_chart.add_point(
                state=state,
                role=role,
                u_p=float(u_p),
                v_p=float(v_p)
            )

        # ============================
        # [ADD] 표 업데이트 로직
        # ============================
        #
        # 요구사항 정리:
        #   - VAC OFF 세션이면:
        #         2열,3열,4열에 main 계측값의 Lv / u' / v'를 쓴다
        #   - VAC ON (또는 보정 이후 세션) 이면:
        #         5열,6열,7열에 main 계측값의 Lv / u' / v'를 쓴다
        #         그리고 8열에는 du'v' = main과 sub의 거리
        #
        #   du'v' = sqrt((u_sub-u_main)^2 + (v_sub-v_main)^2)
        #
        #   row는 현재 cs_idx 인덱스
        #
        #   sub 결과가 없으면 du'v'는 "" 로 둔다.

        # table helper (이미 클래스에 있는 것 그대로 재사용)
        def _safe_set_item(table, r, c, text):
            self._set_item(table, r, c, text if text is not None else "")

        # main/sub 최신 측정값 꺼내기
        # 방금 append 했으므로 store['colorshift'][role][row_idx] 가 방금 결과
        main_ok = row_idx < len(store['colorshift']['main'])
        sub_ok  = row_idx < len(store['colorshift']['sub'])

        main_entry = store['colorshift']['main'][row_idx] if main_ok else (np.nan, np.nan, np.nan, np.nan)
        sub_entry  = store['colorshift']['sub'][row_idx]  if sub_ok  else (np.nan, np.nan, np.nan, np.nan)

        # main_entry = (x, y, u', v')
        x_m, y_m, u_m, v_m = main_entry
        # sub_entry = (x, y, u', v')
        x_s, y_s, u_s, v_s = sub_entry

        # Lv은 results 딕셔너리에서 직접 가져오는 게 더 정확 (store에는 x,y,u',v'만 넣었기 때문)
        # 방금 측정한 results['main'] / results['sub']에서 lv를 다시 읽는다.
        lv_m = np.nan
        lv_s = np.nan
        if 'main' in results and results['main'] is not None:
            _, _, lv_tmp, _, _ = results['main']
            lv_m = float(lv_tmp)
        if 'sub' in results and results['sub'] is not None:
            _, _, lv_tmp2, _, _ = results['sub']
            lv_s = float(lv_tmp2)

        if profile.legend_text.startswith('VAC OFF'):
            # OFF 세션 → 2~4열 업데이트 (row=row_idx)
            #  2열: Lv(main)
            #  3열: u'(main)
            #  4열: v'(main)
            txt_lv = f"{lv_m:.6f}" if np.isfinite(lv_m) else ""
            txt_u  = f"{u_m:.6f}"  if np.isfinite(u_m)  else ""
            txt_v  = f"{v_m:.6f}"  if np.isfinite(v_m)  else ""

            _safe_set_item(tbl_cs_raw, row_idx, 1, txt_lv)
            _safe_set_item(tbl_cs_raw, row_idx, 2, txt_u)
            _safe_set_item(tbl_cs_raw, row_idx, 3, txt_v)

        else:
            # ON 세션 (VAC ON 또는 CORR 이후 세션 포함)
            # 5~7열: Lv(main), u'(main), v'(main)
            txt_lv_on = f"{lv_m:.6f}" if np.isfinite(lv_m) else ""
            txt_u_on  = f"{u_m:.6f}"  if np.isfinite(u_m)  else ""
            txt_v_on  = f"{v_m:.6f}"  if np.isfinite(v_m)  else ""

            _safe_set_item(tbl_cs_raw, row_idx, 4, txt_lv_on)
            _safe_set_item(tbl_cs_raw, row_idx, 5, txt_u_on)
            _safe_set_item(tbl_cs_raw, row_idx, 6, txt_v_on)

            # 8열: du'v' = 거리(main vs sub)
            duv = ""
            if np.isfinite(u_m) and np.isfinite(v_m) and np.isfinite(u_s) and np.isfinite(v_s):
                dist = np.sqrt((u_s - u_m)**2 + (v_s - v_m)**2)
                duv = f"{dist:.6f}"

            _safe_set_item(tbl_cs_raw, row_idx, 7, duv)

이게 지금 표&차트 업데이트 메서드인데, u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y)) 이렇게 계산한 u_p, v_p를 표에 업데이트 하고 이를 통해 델타 u`v`을 다음과 같이 계산하고 싶습니다.
코드 수정 부탁드립니다. 현재는 값이 좀 이상하네요.

델타 u`v` 계산법 = =SQRT(POWER((60도 u`-0도 u`),2)+POWER((60도 v`-0도 v`),2))
