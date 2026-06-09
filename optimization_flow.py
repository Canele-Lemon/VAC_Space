import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator

import src.visualizations.chart_style as cs

class GammaChart:
    _PAT_COLORS = {'white':'black', 'red':'red', 'green':'green', 'blue':'blue'}

    def __init__(self, target_widget, title='Gamma',
                 left=0.10, right=0.95, top=0.95, bottom=0.10,
                 x_tick=64):

        self.fig, (self.ax_main, self.ax_sub) = plt.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)
        self.fig.subplots_adjust(hspace=0.2)

        for i, ax in enumerate((self.ax_main, self.ax_sub)):
            cs.MatFormat_FigArea(ax)

            cs.MatFormat_ChartTitle(ax, title=('Gamma' if i == 0 else None), color='#595959')
            if i == 1:
                cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x', color='#595959', fontsize=9)
            else:
                cs.MatFormat_AxisTitle(ax, axis_title='', axis='x')
            cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y', color='#595959', fontsize=9)

            cs.MatFormat_Axis(
                ax,
                min_val=0,
                max_val=255,
                tick_interval=x_tick,
                axis='x',
                show_labels=(i == 1)
            )

            cs.MatFormat_Axis(
                ax,
                min_val=0.0,
                max_val=1.0,
                tick_interval=None,
                axis='y'
            )

            cs.MatFormat_Gridline(ax, linestyle='--')

        # 시리즈
        lw = 0.8
        self._lines, self._data = {}, {}
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            view_angle = {'main': 'front', 'sub': 'side'}[role]
            
            for pat, col in self._PAT_COLORS.items():
                k_off = ('OFF', role, pat)
                ln_off, = ax.plot([], [], linestyle='--', color=col, linewidth=lw, label=f'OFF {view_angle} {pat}')
                self._lines[k_off] = ln_off; self._data[k_off] = {'x':[], 'y':[]}

                k_on = ('ON', role, pat)
                ln_on, = ax.plot([], [], linestyle='-', color=col, linewidth=lw, label=f'ON {view_angle} {pat}')
                self._lines[k_on] = ln_on; self._data[k_on] = {'x':[], 'y':[]}

        self._update_legends()
        self.canvas.draw_idle()

    def reset_on(self):
        """ON 시리즈만 리셋."""
        for key, ln in self._lines.items():
            if key[0] == 'ON':
                self._data[key]['x'].clear()
                self._data[key]['y'].clear()
                ln.set_data([], [])
        self._autoscale()
        self._update_legends()
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, pattern: str, gray: int, luminance: float):
        key = (state, role, pattern)
        if key not in self._lines:
            return
        self._data[key]['x'].append(int(gray))
        self._data[key]['y'].append(float(luminance))
        self._lines[key].set_data(self._data[key]['x'], self._data[key]['y'])
        self._autoscale(lazy_role=role)
        self._update_legends()
        self.canvas.draw_idle()

    def _autoscale(self, lazy_role=None):
        # 데이터가 1을 넘을 때만 y 상한 확장. 축 갱신도 cs.MatFormat_Axis로 통일.
        roles = [lazy_role] if lazy_role in ('main', 'sub') else ('main', 'sub')
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            if role not in roles:
                continue
            ys = []
            for (state, r, pat), ln in self._lines.items():
                if r == role and len(ln.get_xdata()) and len(ln.get_ydata()):
                    ys.extend(ln.get_ydata())
            ymax = max(ys) if ys else 1.0
            upper = 1.0 if ymax <= 1.0 else ymax * 1.05
            cs.MatFormat_Axis(ax, min_val=0.0, max_val=upper, tick_interval=None,
                              axis='y', tick_color='#bfbfbf', label_color='#595959', label_fontsize=9)

    def _update_legends(self):
        for ax in (self.ax_main, self.ax_sub):
            handles, labels = [], []
            for (state, role, pat), ln in self._lines.items():
                x = ln.get_xdata()
                y = ln.get_ydata()

                if ln.axes is ax and x.size > 0 and y.size > 0:
                    handles.append(ln)
                    labels.append(ln.get_label())

            if handles:
                ax.legend(handles, labels, fontsize=8, loc='upper left')
            else:
                leg = ax.get_legend()
                if leg:
                    leg.remove()

여기서 아래 에러가 발생해요

Traceback (most recent call last):
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 2086, in handle
    self._consume_gamma_pair(pattern, gray, s['_gamma'])
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 2122, in _consume_gamma_pair
    self.vac_optimization_gamma_chart.add_point(
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\charts\gamma_chart.py", line 85, in add_point
    self._update_legends()
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\charts\gamma_chart.py", line 110, in _update_legends
    if ln.axes is ax and x.size > 0 and y.size > 0:
                         ^^^^^^
AttributeError: 'list' object has no attribute 'size'

메인 클래스 코드:

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
        policy = self._spec_policy
        
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        state = 'OFF' if profile.session_mode.startswith('VAC OFF') else 'ON'

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
            self.set_item(table_inst1, gray, cols['lv'], f"{lv_m:.6f}" if np.isfinite(lv_m) else "")
            self.set_item(table_inst1, gray, cols['cx'], f"{cx_m:.6f}" if np.isfinite(cx_m) else "")
            self.set_item(table_inst1, gray, cols['cy'], f"{cy_m:.6f}" if np.isfinite(cy_m) else "")

            # sub 테이블
            lv_s, cx_s, cy_s = store['gamma']['sub']['white'].get(gray, (np.nan, np.nan, np.nan))
            table_inst2 = self.ui.vac_table_opt_mes_results_sub
            self.set_item(table_inst2, gray, cols['lv'], f"{lv_s:.6f}" if np.isfinite(lv_s) else "")
            self.set_item(table_inst2, gray, cols['cx'], f"{cx_s:.6f}" if np.isfinite(cx_s) else "")
            self.set_item(table_inst2, gray, cols['cy'], f"{cy_s:.6f}" if np.isfinite(cy_s) else "")

            # ΔCx/ΔCy (ON 세션에서만; ref_store가 있을 때)                    
            if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
                ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
                if ref_main is not None and np.isfinite(cx_m) and np.isfinite(cy_m):
                    _, cx_r, cy_r = ref_main
                    d_cx = cx_m - cx_r
                    d_cy = cy_m - cy_r

                    if policy.should_eval_color(gray):
                        ok = policy.color_ok(d_cx, d_cy)
                        self.set_item_with_spec(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}", is_spec_ok=ok)
                        self.set_item_with_spec(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}", is_spec_ok=ok)
                    else:
                        self.set_item(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}")
                        self.set_item(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}")

                    # ΔGamma 실시간 계산 및 평가 (VAC OFF max / 현재 ON 0gray 기준)
                    if 'd_gamma' in cols:
                        # 1) OFF 휘도 벡터 (ref_store = VAC OFF)
                        lv_off = np.zeros(256, dtype=np.float64)
                        for gg in range(256):
                            tup_off = profile.ref_store['gamma']['main']['white'].get(gg, None)
                            lv_off[gg] = float(tup_off[0]) if tup_off else np.nan

                        # 2) ON 휘도 벡터 (현재 세션 store)
                        lv_on = np.zeros(256, dtype=np.float64)
                        for gg in range(256):
                            tup_on = store['gamma']['main']['white'].get(gg, None)
                            lv_on[gg] = float(tup_on[0]) if tup_on else np.nan

                        # 3) 정규화 기준: OFF max Lv / ON 0gray Lv
                        Lv_off_max = np.nanmax(lv_off[1:])   # gray 0 제외한 max
                        Lv_on_0    = lv_on[0]

                        if (np.isfinite(Lv_off_max) and np.isfinite(Lv_on_0) and (Lv_off_max > Lv_on_0)):
                            denom = Lv_off_max - Lv_on_0

                            # 정규화된 Y (0~1 근처로 클리핑)
                            Y_off = (lv_off - Lv_on_0) / denom
                            Y_on  = (lv_on  - Lv_on_0) / denom
                            Y_off = np.clip(Y_off, 1e-6, 1-1e-6)
                            Y_on  = np.clip(Y_on,  1e-6, 1-1e-6)

                            # gamma 계산: log(Y) / log(gray_norm)
                            gray_norm = np.linspace(0.0, 1.0, 256, dtype=np.float64)
                            gamma_off = np.full(256, np.nan, dtype=np.float64)
                            gamma_on  = np.full(256, np.nan, dtype=np.float64)

                            valid_off = (gray_norm > 0) & np.isfinite(Y_off)
                            gamma_off[valid_off] = np.log(Y_off[valid_off]) / np.log(gray_norm[valid_off])

                            valid_on = (gray_norm > 0) & np.isfinite(Y_on)
                            gamma_on[valid_on] = np.log(Y_on[valid_on]) / np.log(gray_norm[valid_on])

                            g_off = gamma_off[gray]
                            g_on  = gamma_on[gray]

                            if np.isfinite(g_off) and np.isfinite(g_on):
                                d_gamma = g_on - g_off

                                if policy.should_eval_gamma(gray):
                                    ok = policy.gamma_ok(d_gamma)
                                    self.set_item_with_spec(table_inst1, gray, cols['d_gamma'], f"{d_gamma:.6f}", is_spec_ok=ok)
                                else:
                                    self.set_item(table_inst1, gray, cols['d_gamma'], f"{d_gamma:.6f}")



