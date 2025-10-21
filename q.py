class GammaChartVAC:
    """
    두 축(0° main / 60° sub), OFF는 참조로 유지, ON은 런마다 reset_on().
    CIE1976 차트와 동일한 서식(cs.MatFormat_*) 사용.
    """
    _PAT_COLORS = {'white':'gray', 'red':'red', 'green':'green', 'blue':'blue'}

    def __init__(self, target_widget, title='Gamma',
                 left=0.10, right=0.95, top=0.95, bottom=0.10,
                 x_tick=64, y_tick=None):
        # 두 개 축 (세로), x 공유
        self.fig, (self.ax_main, self.ax_sub) = plt.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # ── 공통 서식: CIE1976과 동일한 cs.* 사용 ──
        cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
        for i, (ax, atitle) in enumerate(((self.ax_main, 'Gamma (Main 0°)'),
                                          (self.ax_sub,  'Gamma (Sub 60°)'))):
            cs.MatFormat_FigArea(ax)
            # 상단/하단 축 공통 타이틀 스타일
            cs.MatFormat_ChartTitle(ax, title=atitle, color='#595959')
            # x축 제목은 하단 축에만
            if i == 1:
                cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x')
            else:
                cs.MatFormat_AxisTitle(ax, axis_title='', axis='x')  # 숨김
            cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y')

            # 축 범위/눈금: Gray 0..255 / Lv 자동. x_tick(기본 64)
            cs.MatFormat_Axis(ax, min_val=0, max_val=255, tick_interval=x_tick, axis='x')
            # y축 눈금 간격 지정 없으면 자동; 지정하고 싶으면 y_tick 값 전달
            if y_tick is not None:
                # 초기 높이는 1로 가정; 데이터 들어오면 자동으로 autoscale
                cs.MatFormat_Axis(ax, min_val=0, max_val=1, tick_interval=y_tick, axis='y')
            else:
                cs.MatFormat_Axis(ax, min_val=0, max_val=1, tick_interval=0.25, axis='y')

            cs.MatFormat_Gridline(ax, linestyle='--')

        # ── 시리즈: OFF/ON × main/sub × 패턴 ──
        self._lines = {}
        self._data  = {}
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            for pat, col in self._PAT_COLORS.items():
                # OFF(점선)
                k_off = ('OFF', role, pat)
                ln_off, = ax.plot([], [], linestyle='--', color=col, label=f'OFF {role} {pat}')
                self._lines[k_off] = ln_off; self._data[k_off] = {'x':[], 'y':[]}
                # ON(실선)
                k_on = ('ON', role, pat)
                ln_on, = ax.plot([], [], linestyle='-', color=col, label=f'ON {role} {pat}')
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
        self.canvas.draw_idle()

    # ── 내부 유틸 ──
    def _autoscale(self, lazy_role=None):
        roles = [lazy_role] if lazy_role in ('main', 'sub') else ('main', 'sub')
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            if role not in roles: continue
            ys = []
            for (state, r, pat), ln in self._lines.items():
                if r==role and ln.get_xdata() and ln.get_ydata():
                    ys.extend(ln.get_ydata())
            if ys:
                ymax = max(ys)
                ax.set_ylim(0, max(1e-6, ymax)*1.05)

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