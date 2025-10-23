감마y축 tick안쪽으로, 색깔 변경, y축 폰트 색깔 및 크기(cs. 차트 서식으로), 처음 초기화 때 0~1 레인지로 해 주세요.

현재 클래스는 아래와 같아요.

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import modules.chart_style as cs
from matplotlib.ticker import MaxNLocator

class GammaChart:
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
        self.fig.subplots_adjust(hspace=0.2)

        # ── 공통 서식: CIE1976과 동일한 cs.* 사용 ──
        # cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
        for i, (ax, atitle) in enumerate(((self.ax_main, 'Gamma'), (self.ax_sub,  'Gamma'))):
            cs.MatFormat_FigArea(ax)
            # (1) 제목 표시: 위쪽 축만
            if i == 0:
                cs.MatFormat_ChartTitle(ax, title=atitle, color='#595959')
            else:
                cs.MatFormat_ChartTitle(ax, title=None)  # 아래쪽 제목 제거

            # (2) x축 제목 및 눈금: 아래쪽 축만
            if i == 1:
                cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x')
                ax.tick_params(axis='x', which='both', labelbottom=True)  # 눈금 표시
            else:
                cs.MatFormat_AxisTitle(ax, axis_title='', axis='x')  # 숨김
                ax.tick_params(axis='x', which='both', labelbottom=False)  # 눈금 숨김
                
            # (3) x축 범위/틱만 고정, y축은 오토스케일
            cs.MatFormat_Axis(ax, min_val=0, max_val=255, tick_interval=x_tick, axis='x')
            # 🔽 y축은 범위 미고정: 눈금만 5개 내로 자동 배치
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y')
            cs.MatFormat_Gridline(ax, linestyle='--')

        # ── 시리즈: OFF/ON × main/sub × 패턴 ──
        lw = 0.8
        self._lines = {}; self._data  = {}
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            for pat, col in self._PAT_COLORS.items():
                # OFF(점선)
                k_off = ('OFF', role, pat)
                ln_off, = ax.plot([], [], linestyle='--', color=col, linewidth=lw, label=f'OFF {role} {pat}')
                self._lines[k_off] = ln_off; self._data[k_off] = {'x':[], 'y':[]}
                # ON(실선)
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
