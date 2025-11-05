    def start_viewing_angle_session(self,
        profile: SessionProfile,
        gray_levels=None,
        gamma_patterns=('white','red','green','blue'),
        colorshift_patterns=None,
        first_gray_delay_ms=3000,
        gamma_settle_ms=1000,
        cs_settle_ms=1000,
        on_done=None,
    ):
        if gray_levels is None:
            gray_levels = op.gray_levels_256
        if colorshift_patterns is None:
            colorshift_patterns = op.colorshift_patterns
        
        gamma_patterns=('white',)
        store = {
            'gamma': {'main': {p:{} for p in gamma_patterns}, 'sub': {p:{} for p in gamma_patterns}},
            'colorshift': {'main': [], 'sub': []}
        }

        self._sess = {
            'phase': 'gamma',
            'p_idx': 0,
            'g_idx': 0,
            'cs_idx': 0,
            'patterns': list(gamma_patterns),
            'gray_levels': list(gray_levels),
            'cs_patterns': colorshift_patterns,
            'store': store,
            'profile': profile,
            'first_gray_delay_ms': first_gray_delay_ms,
            'gamma_settle_ms': gamma_settle_ms,
            'cs_settle_ms': cs_settle_ms,
            'on_done': on_done
        }
        self._session_step()

    def _session_step(self):
        s = self._sess
        if s.get('paused', False):
            return
        
        if s['phase'] == 'gamma':
            if s['p_idx'] >= len(s['patterns']):
                s['phase'] = 'colorshift'
                s['cs_idx'] = 0
                QTimer.singleShot(60, lambda: self._session_step())
                return

            if s['g_idx'] >= len(s['gray_levels']):
                s['g_idx'] = 0
                s['p_idx'] += 1
                QTimer.singleShot(40, lambda: self._session_step())
                return

            pattern = s['patterns'][s['p_idx']]
            gray = s['gray_levels'][s['g_idx']]

            if pattern == 'white':
                rgb_value = f"{gray},{gray},{gray}"
            elif pattern == 'red':
                rgb_value = f"{gray},0,0"
            elif pattern == 'green':
                rgb_value = f"0,{gray},0"
            else:
                rgb_value = f"0,0,{gray}"
            self.changeColor(rgb_value)

            if s['g_idx'] == 0:
                delay = s['first_gray_delay_ms']
            else:
                delay = s.get('gamma_settle_ms', 0)
            QTimer.singleShot(delay, lambda p=pattern, g=gray: self._trigger_gamma_pair(p, g))

        elif s['phase'] == 'colorshift':
            if s['cs_idx'] >= len(s['cs_patterns']):
                s['phase'] = 'done'
                QTimer.singleShot(0, lambda: self._session_step())
                return

            pname, r, g, b = s['cs_patterns'][s['cs_idx']]
            self.changeColor(f"{r},{g},{b}")
            QTimer.singleShot(s['cs_settle_ms'], lambda pn=pname: self._trigger_colorshift_pair(pn))

        else:  # done
            self._finalize_session()

    def _trigger_gamma_pair(self, pattern, gray):
        s = self._sess
        s['_gamma'] = {}

        def handle(role, res):
            s['_gamma'][role] = res
            got_main = 'main' in s['_gamma']
            got_sub = ('sub') in s['_gamma'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_gamma_pair(pattern, gray, s['_gamma'])
                
                if s.get('paused', False):
                    return
                
                s['g_idx'] += 1
                QTimer.singleShot(30, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle)
            self.sub_measure_thread.start()

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

        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
                continue

            x, y, lv, cct, duv = res
            store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

            self.vac_optimization_gamma_chart.add_point(
                state=state,
                role=role,               # 'main'/'sub'
                pattern=pattern,         # 'white'/'red'/'green'/'blue'
                gray=int(gray),
                luminance=float(lv)
            )

        if pattern == 'white':
            is_on_session = (profile.ref_store is not None)
            is_fine_mode = getattr(self, "_fine_mode", False)

            if is_on_session:
                ref_store = profile.ref_store
                # main role 기준으로 0gray 휘도 사용
                lv0_main, _, _ = store['gamma']['main']['white'].get(0, (np.nan, np.nan, np.nan))
                if np.isfinite(lv0_main):
                    self._on_lv0_current = float(lv0_main)
            
            if is_on_session and is_fine_mode:
                ok_now = self._is_gray_spec_ok(gray, thr_gamma=0.05, thr_c=0.003, off_store=self._off_store, on_store=s['store'])
                
                if not ok_now and not self._sess.get('paused', False):
                    logging.info(f"[Fine Correction] gray={gray} NG → per-gray correction start")
                    self._start_gray_ng_correction(gray, max_retries=3, thr_gamma=0.05, thr_c=0.003)
                    
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

    def _trigger_colorshift_pair(self, patch_name):
        s = self._sess
        s['_cs'] = {}

        def handle(role, res):
            s['_cs'][role] = res
            got_main = 'main' in s['_cs']
            got_sub = ('sub') in s['_cs'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_colorshift_pair(patch_name, s['_cs'])
                s['cs_idx'] += 1
                QTimer.singleShot(80, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle)
            self.sub_measure_thread.start()

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
