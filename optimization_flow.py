    def _update_legend(self):
        handles, labels = [], []

        for ln in self.ax.lines:
            lb = ln.get_label()
            if lb in ("BT.709", "DCI"):
                handles.append(ln)
                labels.append(lb)

        for k in (('OFF', 'main'), ('OFF', 'sub'), ('ON', 'main'), ('ON', 'sub')):
            ln = self.lines.get(k)

            if ln is not None:
                x = ln.get_xdata()
                y = ln.get_ydata()

                if x.size > 0 and y.size > 0:
                    handles.append(ln)
                    labels.append(ln.get_label())

        if handles:
            self.ax.legend(handles, labels, fontsize=8, loc='lower right')
        else:
            leg = self.ax.get_legend()
            if leg:
                leg.remove()

여기서도 아래 에러 발생:
Traceback (most recent call last):
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 2239, in handle
    if got_main and got_sub:
        ^^^^^^^^^^^^^^^^^^^^^
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 2297, in _consume_colorshift_pair
    # 차트 갱신 (vac_optimization_cie1976_chart 는 u' v' scatter)
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\charts\chromaticity_diagram.py", line 107, in add_point
    self._update_legend()
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\charts\chromaticity_diagram.py", line 127, in _update_legend
    if x.size > 0 and y.size > 0:
       ^^^^^^
AttributeError: 'list' object has no attribute 'size'

메인 클래스 코드:

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
        state = 'OFF' if profile.session_mode.startswith('VAC OFF') else 'ON'

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
            u_p, v_p = op.convert_xyz_to_uvprime(float(x), float(y))

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

        if profile.session_mode.startswith('VAC OFF'):
            # ---------- VAC OFF ----------
            # row_idx 행의
            #   col=1 → Lv(main)
            #   col=2 → u'(main)
            #   col=3 → v'(main)

            txt_lv_off = f"{lv_main:.6f}" if np.isfinite(lv_main) else ""
            txt_u_off  = f"{up_main:.6f}"  if np.isfinite(up_main)  else ""
            txt_v_off  = f"{vp_main:.6f}"  if np.isfinite(vp_main)  else ""

            self.set_item(tbl_cs_raw, row_idx, 1, txt_lv_off)
            self.set_item(tbl_cs_raw, row_idx, 2, txt_u_off)
            self.set_item(tbl_cs_raw, row_idx, 3, txt_v_off)

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

            self.set_item(tbl_cs_raw, row_idx, 4, txt_lv_on)
            self.set_item(tbl_cs_raw, row_idx, 5, txt_u_on)
            self.set_item(tbl_cs_raw, row_idx, 6, txt_v_on)

            # du'v' 계산
            # 엑셀식: =SQRT( (60deg_u' - 0deg_u')^2 + (60deg_v' - 0deg_v')^2 )
            # 여기서 main=0°, sub=60°
            duv_txt = ""
            if np.isfinite(up_main) and np.isfinite(vp_main) and np.isfinite(up_sub) and np.isfinite(vp_sub):
                dist = np.sqrt((up_sub - up_main)**2 + (vp_sub - vp_main)**2)
                duv_txt = f"{dist:.6f}"

            self.set_item(tbl_cs_raw, row_idx, 7, duv_txt)
            
    def add_point(self, *, state: str, role: str, u_p: float, v_p: float):
        key = (state, role)
        role_map = {'main': 'front', 'sub': 'side'}
        view_angle = role_map.get(role, role)

        if key not in self.lines:
            return
        self.data[key]['u'].append(float(u_p))
        self.data[key]['v'].append(float(v_p))
        self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
        self.lines[key].set_label(f"{state} {view_angle}")
        self._update_legend()
        self.canvas.draw_idle()

        
