class BarChart:
    def __init__(self, target_widget, title='Grouped Bars',
                 x_labels=None, y_label='Value',
                 y_range=(0, 0.08), y_tick=0.02,
                 series_labels=('VAC OFF','VAC ON'),
                 spec_line=None):
        self.x_labels = x_labels or []
        self.y_label = y_label
        self.y_range = y_range
        self.y_tick = y_tick
        self.series_labels = series_labels
        self.spec_line = spec_line

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        cs.MatFormat_ChartArea(self.fig, left=0.12, right=0.96, top=0.9, bottom=0.18)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title)
        cs.MatFormat_AxisTitle(self.ax, self.y_label, axis='y')
        cs.MatFormat_Axis(self.ax, self.y_range[0], self.y_range[1], self.y_tick, axis='y')
        self.ax.set_xticks(np.arange(len(self.x_labels)))
        self.ax.set_xticklabels(self.x_labels, fontsize=9, color='#595959', rotation=0)
        self.ax.tick_params(axis='x', length=0)
        self.ax.legend(fontsize=8)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._bars = None
        self.canvas.draw()

    def update_grouped(self, data_off, data_on):
        """data_off/on: 길이=len(x_labels)"""
        self.ax.clear()
        # 스타일 재적용
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, self.ax.get_title())
        cs.MatFormat_AxisTitle(self.ax, self.y_label, axis='y')
        cs.MatFormat_Axis(self.ax, self.y_range[0], self.y_range[1], self.y_tick, axis='y')
        x = np.arange(len(self.x_labels))
        width = 0.38
        b1 = self.ax.bar(x - width/2, data_off, width, label=self.series_labels[0])
        b2 = self.ax.bar(x + width/2, data_on,  width, label=self.series_labels[1])
        if self.spec_line is not None:
            self.ax.axhline(self.spec_line, linestyle='--', linewidth=0.8, color='red')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.x_labels, fontsize=9, color='#595959')
        self.ax.legend(fontsize=8, loc='upper right')
        self.canvas.draw()

class GammaChart:
    _PAT_COLORS = {'white':'gray', 'red':'red', 'green':'green', 'blue':'blue'}

    def __init__(self, target_widget, title='Gamma',
                 left=0.10, right=0.95, top=0.95, bottom=0.10,
                 x_tick=64):
        from modules import chart_style as cs
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        self.fig, (self.ax_main, self.ax_sub) = plt.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)
        self.fig.subplots_adjust(hspace=0.2)

        # 공통 서식 (cs 모듈만 사용)
        for i, ax in enumerate((self.ax_main, self.ax_sub)):
            cs.MatFormat_FigArea(ax)

            # 제목/축제목
            cs.MatFormat_ChartTitle(ax, title=('Gamma' if i == 0 else None), color='#595959')
            if i == 1:
                cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x', color='#595959', fontsize=9)
            else:
                cs.MatFormat_AxisTitle(ax, axis_title='', axis='x', show_labels=False)
            cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y', color='#595959', fontsize=9)

            # 축 범위/틱: x는 0~255, y는 0~1로 초기화
            cs.MatFormat_Axis(ax, min_val=0,   max_val=255, tick_interval=x_tick,
                              axis='x', tick_color='#bfbfbf', label_color='#595959', label_fontsize=9)
            cs.MatFormat_Axis(ax, min_val=0.0, max_val=1.0,  tick_interval=None,
                              axis='y', tick_color='#bfbfbf', label_color='#595959', label_fontsize=9)

            cs.MatFormat_Gridline(ax, linestyle='--')

        # 시리즈
        lw = 0.8
        self._lines, self._data = {}, {}
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            for pat, col in self._PAT_COLORS.items():
                k_off = ('OFF', role, pat)
                ln_off, = ax.plot([], [], linestyle='--', color=col, linewidth=lw, label=f'OFF {role} {pat}')
                self._lines[k_off] = ln_off; self._data[k_off] = {'x':[], 'y':[]}

                k_on = ('ON', role, pat)
                ln_on, = ax.plot([], [], linestyle='-', color=col, linewidth=lw, label=f'ON {role} {pat}')
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
        from modules import chart_style as cs
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
                if ln.axes is ax and ln.get_xdata() and ln.get_ydata():
                    handles.append(ln); labels.append(ln.get_label())
            if handles:
                ax.legend(handles, labels, fontsize=8, loc='upper left')
            else:
                leg = ax.get_legend()
                if leg: leg.remove()

class CIE1976Chart:
    """
    - state: 'OFF' (레퍼런스, 빨강) / 'ON' (최적화/보정, 초록)
    - role:  'main' (0°) / 'sub' (60°)  → 마커: main=o, sub=s (hollow)
    - 배경 이미지/BT.709/DCI/CIE1976 등온선은 항상 표시
    - reset_on(): 'ON' 시리즈만 리셋
    - add_point(state, role, u_p, v_p): 한 점 추가 (각 세션/그레이별 누적)
    """
    def __init__(self, target_widget, title="Color Shift",
                 left_margin=0.10, right_margin=0.95, top_margin=0.95, bottom_margin=0.10):

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # ── 배경 이미지 ──
        try:
            image_path = cf.get_normalized_path(
                __file__, '..','..','..', 'resources/images/pictures', 'cie1976 (2).png'
            )
            if os.path.exists(image_path):
                img = plt.imread(image_path, format='png')
                self.ax.imshow(img, extent=[0, 0.70, 0, 0.60])
        except Exception as e:
            print(f"[CIE1976] BG load fail: {e}")

        # ── 서식(기존과 동일) ──
        # cs.MatFormat_ChartArea(self.fig, left=left_margin, right=right_margin,top=top_margin, bottom=bottom_margin)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title=title, color='#595959')
        cs.MatFormat_AxisTitle(self.ax, axis_title="u`", axis='x')
        cs.MatFormat_AxisTitle(self.ax, axis_title="v`", axis='y')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.7, tick_interval=0.1, axis='x')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.6, tick_interval=0.1, axis='y')
        cs.MatFormat_Gridline(self.ax, linestyle='--')

        # ── 레퍼런스 경계 ──
        try:
            BT709_u, BT709_v = cf.convert2DlistToPlot(op.BT709_uvprime)
            self.ax.plot(BT709_u, BT709_v, color='black', linestyle='--', linewidth=0.8, label="BT.709")
        except Exception as e:
            print(f"[CIE1976] BT.709 plot fail: {e}")
        try:
            DCI_u, DCI_v = cf.convert2DlistToPlot(op.DCI_uvprime)
            self.ax.plot(DCI_u, DCI_v, color='black', linestyle='-', linewidth=0.8, label="DCI")
        except Exception as e:
            print(f"[CIE1976] DCI plot fail: {e}")
        try:
            CIE1976_u = [r[1] for r in op.CIE1976_uvprime]
            CIE1976_v = [r[2] for r in op.CIE1976_uvprime]
            self.ax.plot(CIE1976_u, CIE1976_v, color='black', linestyle='-', linewidth=0.3)  # label=None
        except Exception as e:
            print(f"[CIE1976] iso plot fail: {e}")

        # ── 데이터 시리즈: (state, role) ──
        # 색: OFF=red, ON=green / 마커: main='o', sub='s' / hollow
        ms = 3.5
        self.lines = {
            ('OFF', 'main'): self.ax.plot([], [], 'o',
                                          markersize=ms,
                                          markerfacecolor='none',
                                          markeredgecolor='red', linewidth=0)[0],
            ('OFF', 'sub'):  self.ax.plot([], [], 'o',
                                          markersize=ms,
                                          markerfacecolor='none',
                                          markeredgecolor='green', linewidth=0)[0],
            ('ON', 'main'):  self.ax.plot([], [], 'o',
                                          markersize=ms,
                                          markerfacecolor='red',
                                          markeredgecolor='red', linewidth=0)[0],
            ('ON', 'sub'):   self.ax.plot([], [], 'o',
                                          markersize=ms,
                                          markerfacecolor='green',
                                          markeredgecolor='green', linewidth=0)[0],
        }
        self.data = {k: {'u': [], 'v': []} for k in self.lines.keys()}
        self._update_legend()

    # ── public API ──
    def reset_on(self):
        """ON(보정/적용) 시리즈만 초기화; OFF(레퍼런스)는 유지."""
        for k in (('ON', 'main'), ('ON', 'sub')):
            self.data[k]['u'].clear()
            self.data[k]['v'].clear()
            self.lines[k].set_data([], [])
        self._update_legend()
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, u_p: float, v_p: float):
        """
        사용처 예:
          self.vac_optimization_cie1976_chart.add_point(
              state=('OFF' or 'ON'), role=('main' or 'sub'), u_p=..., v_p=...
          )
        """
        key = (state, role)
        if key not in self.lines:
            return
        self.data[key]['u'].append(float(u_p))
        self.data[key]['v'].append(float(v_p))
        self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
        self.lines[key].set_label(f"{state} {role}")
        self._update_legend()
        self.canvas.draw_idle()

    # ── internals ──
    def _update_legend(self):
        handles, labels = [], []
        for ln in self.ax.lines:
            lb = ln.get_label()
            if lb in ("BT.709", "DCI"):
                handles.append(ln)
                labels.append(lb)
        for k in (('OFF', 'main'), ('OFF', 'sub'), ('ON', 'main'), ('ON', 'sub')):
            ln = self.lines.get(k)
            if ln and ln.get_xdata() and ln.get_ydata():
                handles.append(ln)
                labels.append(ln.get_label())
        if handles:
            self.ax.legend(handles, labels, fontsize=8, loc='lower right')
        else:
            leg = self.ax.get_legend()
            if leg:
                leg.remove()

class LUTChart:
    def __init__(self, target_widget, title='TV LUT (12-bit)',
                 left=0.10, right=0.95, top=0.95, bottom=0.10,
                 x_tick=512, y_tick=512):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # ── CIE와 동일한 포맷 적용 ──
        # cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title=title, color='#595959')
        cs.MatFormat_AxisTitle(self.ax, axis_title='Gray Level (12-bit)', axis='x')
        cs.MatFormat_AxisTitle(self.ax, axis_title='Input Level', axis='y')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=4095, tick_interval=x_tick, axis='x')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=4095, tick_interval=y_tick, axis='y')
        cs.MatFormat_Gridline(self.ax, linestyle='--')

        self._lines = {}
        self.canvas.draw_idle()

    def reset_and_plot(self, lut_dict: dict):
        """
        lut_dict = {
          "R_Low":[4096], "R_High":[4096],
          "G_Low":[4096], "G_High":[4096],
          "B_Low":[4096], "B_High":[4096],
        }
        """
        # 기존 라인 제거
        for ln in list(self._lines.values()):
            try: ln.remove()
            except Exception: pass
        self._lines.clear()

        xs = np.arange(4096)
        styles = {
            'R_Low':  dict(color='red',   ls='--', label='R Low'),
            'R_High': dict(color='red',   ls='-',  label='R High'),
            'G_Low':  dict(color='green', ls='--', label='G Low'),
            'G_High': dict(color='green', ls='-',  label='G High'),
            'B_Low':  dict(color='blue',  ls='--', label='B Low'),
            'B_High': dict(color='blue',  ls='-',  label='B High'),
        }
        ymax = 0.0
        for k, st in styles.items():
            ys = np.asarray(lut_dict.get(k, []), dtype=float).ravel()
            if ys.size != 4096:
                print(f"[LUTChartVAC] {k} length invalid: {ys.size}")
                continue
            ln, = self.ax.plot(xs, ys, **st)
            self._lines[k] = ln
            if ys.size:
                m = np.nanmax(ys)
                if np.isfinite(m): ymax = max(ymax, float(m))

        # 축/범례 갱신
        self.ax.set_xlim(0, 4095)
        self.ax.set_ylim(0, max(4095.0, ymax*1.05 if ymax>0 else 4095.0))
        if self._lines:
            self.ax.legend([ln for ln in self._lines.values()],
                           [ln.get_label() for ln in self._lines.values()],
                           fontsize=8, loc='upper left')
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()

        self.canvas.draw_idle()

class XYChart:
    def __init__(self, target_widget, x_label='X', y_label='Y',
                 x_range=(0, 100), y_range=(0, 100), x_tick=10, y_tick=10,
                 title=None, title_color='#333333', legend=True,
                 multi_axes=False, num_axes=2, layout='vertical', share_x=True):
        
        self.multi_axes = multi_axes
        self.lines = {}
        self.data = {}
        
        if self.multi_axes:
            if layout == 'vertical':
                self.fig, axes = plt.subplots(num_axes, 1, sharex=share_x)
            else:
                self.fig, axes = plt.subplots(1, num_axes, sharex=share_x)   

            self.axes = list(axes) if isinstance(axes, (list, tuple, np.ndarray)) else [axes]
            self.ax = self.axes[0]  # 기본 축
        else:
            self.fig, self.ax = plt.subplots()
            self.axes = [self.ax]

        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # 스타일 초기화
        self._init_style(title, title_color, x_label, y_label, x_range, y_range, x_tick, y_tick)

        if legend:
            for ax in self.axes:
                cs.MatFormat_Legend(ax, position='upper left', fontsize=8)

        self.canvas.draw()

    def _init_style(self, title, title_color, x_label, y_label, x_range, y_range, x_tick, y_tick):
        cs.MatFormat_ChartArea(self.fig, left=0.20, right=0.92, top=0.90, bottom=0.15)
        for i, ax in enumerate(self.axes):
            cs.MatFormat_FigArea(ax)
            if i == 0:
                cs.MatFormat_ChartTitle(ax, title=title, color=title_color)

            cs.MatFormat_AxisTitle(ax, axis_title=x_label, axis='x', show_labels=(i == len(self.axes) - 1))
            cs.MatFormat_AxisTitle(ax, axis_title=y_label, axis='y')
            cs.MatFormat_Axis(ax, min_val=x_range[0], max_val=x_range[1], tick_interval=x_tick, axis='x')
            cs.MatFormat_Axis(ax, min_val=y_range[0], max_val=y_range[1], tick_interval=y_tick, axis='y')
            cs.MatFormat_Gridline(ax)

    def add_line(self, key, color='blue', linestyle='-', marker=None, label=None, axis_index=0):
        if axis_index >= len(self.axes):
            print(f"[XYChart] Invalid axis index: {axis_index}")
            return

        axis = self.axes[axis_index]
        line, = axis.plot([], [], color=color, linestyle=linestyle, marker=marker, label=label or key)
        self.lines[key] = line
        self.data[key] = {'x': [], 'y': []}

    def update(self, key, x, y):
        if key not in self.lines:
            print(f"[XYChart] Line '{key}' not found.")
            return

        self.data[key]['x'].append(x)
        self.data[key]['y'].append(y)
        self.lines[key].set_data(self.data[key]['x'], self.data[key]['y'])

        axis = self.lines[key].axes
        axis.relim()
        axis.autoscale_view()
        axis.legend(fontsize=9)
        self.canvas.draw()

    def set_label(self, key, label):
        if key in self.lines:
            self.lines[key].set_label(label)
            self.lines[key].axes.legend(fontsize=9)
            
    def update_legend(self):
        for ax in self.axes:
            handles, labels = ax.get_legend_handles_labels()

            def _has_data(h):
                # Line2D가 아닐 수도 있으니 방어적으로
                getx = getattr(h, "get_xdata", None)
                gety = getattr(h, "get_ydata", None)
                if callable(getx) and callable(gety):
                    x = np.asarray(getx())
                    y = np.asarray(gety())
                    return x.size > 0 and y.size > 0
                return False

            filtered_pairs = [(h, l) for h, l in zip(handles, labels) if _has_data(h)]
            leg = ax.get_legend()

            if filtered_pairs:
                fh, fl = zip(*filtered_pairs)
                ax.legend(fh, fl, loc='upper right', fontsize=8)
            else:
                if leg is not None:
                    leg.remove()
                
    def set_series(self, key, x_list, y_list, *, marker=None, linestyle='-', label=None, axis_index=0):
        """한 번에 x/y 전체를 세팅 (배치 갱신)"""
        # 1) 1D 배열로 강제 + 길이 체크
        x_arr = np.asarray(x_list).reshape(-1)
        y_arr = np.asarray(y_list).reshape(-1)
        if x_arr.size != y_arr.size:
            print(f"[XYChart] set_series: length mismatch (x={x_arr.size}, y={y_arr.size})")
            return

        # 2) 라인 생성(없으면)
        if key not in self.lines:
            axis = self.axes[axis_index]
            line, = axis.plot([], [], linestyle=linestyle, marker=marker, label=label or key)
            self.lines[key] = line
            self.data[key] = {'x': [], 'y': []}

        # 3) 데이터 반영 (list→array여도 OK)
        self.data[key]['x'] = x_arr.tolist()
        self.data[key]['y'] = y_arr.tolist()

        ln = self.lines[key]
        if marker is not None: ln.set_marker(marker)
        ln.set_linestyle(linestyle)
        if label is not None: ln.set_label(label)
        ln.set_data(self.data[key]['x'], self.data[key]['y'])

        ax = ln.axes
        ax.relim(); ax.autoscale_view()
        self.update_legend()
        self.canvas.draw()

    def clear_series(self, key=None):
        """특정 시리즈만(또는 전체) 클리어"""
        if key is None:
            for k in list(self.lines.keys()):
                self.clear_series(k)
            return
        if key in self.lines:
            self.data[key]['x'].clear()
            self.data[key]['y'].clear()
            self.lines[key].set_data([], [])
            self.update_legend()
            self.canvas.draw()
