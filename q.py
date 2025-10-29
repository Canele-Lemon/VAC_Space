def _consume_colorshift_pair(self, patch_name, results):
    """
    results: {
        'main': (x, y, lv, cct, duv)  또는  None,   # main = 0°
        'sub' : (x, y, lv, cct, duv)  또는  None    # sub  = 60°
    }
    """
    s = self._sess
    store = s['store']
    profile: SessionProfile = s['profile']

    # 현재 세션 상태 문자열 ('VAC OFF...' 이면 OFF, 아니면 ON)
    state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

    # 이 측정 패턴의 row index (op.colorshift_patterns 순서 그대로)
    row_idx = s['cs_idx']

    # 이 테이블: vac_table_opt_mes_results_colorshift
    tbl_cs_raw = self.ui.vac_table_opt_mes_results_colorshift

    # ------------------------------------------------
    # 1) main / sub 결과 변환해서 store에 넣고 차트 갱신
    #    store['colorshift'][role][row_idx] = (Lv, u', v')
    # ------------------------------------------------
    for role in ('main', 'sub'):
        res = results.get(role, None)
        if res is None:
            # 측정 실패 시 해당 row에 placeholder 저장
            store['colorshift'][role].append((np.nan, np.nan, np.nan))
            continue

        x, y, lv, cct, duv_unused = res

        # xy -> u' v'
        u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))

        # store에 (Lv, u', v') 저장
        store['colorshift'][role].append((
            float(lv),
            float(u_p),
            float(v_p),
        ))

        # 차트 갱신 (vac_optimization_cie1976_chart 는 u' v' scatter)
        self.vac_optimization_cie1976_chart.add_point(
            state=state,
            role=role,      # 'main' or 'sub'
            u_p=float(u_p),
            v_p=float(v_p)
        )

    # ------------------------------------------------
    # 2) 표 업데이트
    #    OFF 세션:
    #        2열,3열,4열 ← main의 Lv / u' / v'
    #    ON/CORR 세션:
    #        5열,6열,7열 ← main의 Lv / u' / v'
    #        8열        ← du'v' (sub vs main 거리)
    # ------------------------------------------------

    # 이제 방금 append한 값들을 row_idx에서 꺼냄
    main_ok = row_idx < len(store['colorshift']['main'])
    sub_ok  = row_idx < len(store['colorshift']['sub'])

    if main_ok:
        lv_main, up_main, vp_main = store['colorshift']['main'][row_idx]
    else:
        lv_main, up_main, vp_main = (np.nan, np.nan, np.nan)

    if sub_ok:
        lv_sub, up_sub, vp_sub = store['colorshift']['sub'][row_idx]
    else:
        lv_sub, up_sub, vp_sub = (np.nan, np.nan, np.nan)

    # 테이블에 안전하게 set 하는 helper
    def _safe_set_item(table, r, c, text):
        self._set_item(table, r, c, text if text is not None else "")

    if profile.legend_text.startswith('VAC OFF'):
        # ---------- VAC OFF ----------
        # row_idx 행의
        #   col=1 → Lv(main)
        #   col=2 → u'(main)
        #   col=3 → v'(main)

        txt_lv_off = f"{lv_main:.6f}" if np.isfinite(lv_main) else ""
        txt_u_off  = f"{up_main:.6f}"  if np.isfinite(up_main)  else ""
        txt_v_off  = f"{vp_main:.6f}"  if np.isfinite(vp_main)  else ""

        _safe_set_item(tbl_cs_raw, row_idx, 1, txt_lv_off)
        _safe_set_item(tbl_cs_raw, row_idx, 2, txt_u_off)
        _safe_set_item(tbl_cs_raw, row_idx, 3, txt_v_off)

    else:
        # ---------- VAC ON (또는 CORR 이후) ----------
        # row_idx 행의
        #   col=4 → Lv(main)
        #   col=5 → u'(main)
        #   col=6 → v'(main)
        #   col=7 → du'v' = sqrt((u'_sub - u'_main)^2 + (v'_sub - v'_main)^2)

        txt_lv_on = f"{lv_main:.6f}" if np.isfinite(lv_main) else ""
        txt_u_on  = f"{up_main:.6f}"  if np.isfinite(up_main)  else ""
        txt_v_on  = f"{vp_main:.6f}"  if np.isfinite(vp_main)  else ""

        _safe_set_item(tbl_cs_raw, row_idx, 4, txt_lv_on)
        _safe_set_item(tbl_cs_raw, row_idx, 5, txt_u_on)
        _safe_set_item(tbl_cs_raw, row_idx, 6, txt_v_on)

        # du'v' 계산
        # 엑셀식: =SQRT( (60deg_u' - 0deg_u')^2 + (60deg_v' - 0deg_v')^2 )
        # 여기서 main=0°, sub=60°
        duv_txt = ""
        if np.isfinite(up_main) and np.isfinite(vp_main) and np.isfinite(up_sub) and np.isfinite(vp_sub):
            dist = np.sqrt((up_sub - up_main)**2 + (vp_sub - vp_main)**2)
            duv_txt = f"{dist:.6f}"

        _safe_set_item(tbl_cs_raw, row_idx, 7, duv_txt)