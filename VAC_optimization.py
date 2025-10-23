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