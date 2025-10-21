# ─────────────────────────────────────────────────────────────────────────
# GammaChartVAC: OFF는 참조로 고정, ON은 매 회차 리셋되는 단순 차트
# ─────────────────────────────────────────────────────────────────────────
class GammaChartVAC:
    """
    두 축 (main/sub). 패턴은 white/red/green/blue 지원.
    - OFF(Ref) 시리즈: 항상 유지 (점진적으로 누적)
    - ON 시리즈: 매 'ON 측정 시작' 또는 '보정 후 재측정 시작' 시점에 reset_on()
    """
    _PAT_COLORS = {'white':'gray', 'red':'red', 'green':'green', 'blue':'blue'}

    def __init__(self, target_widget):
        self.fig, (self.ax_main, self.ax_sub) = plt.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # 기본 스타일
        for ax, title in ((self.ax_main, 'Gamma (Main 0°)'), (self.ax_sub, 'Gamma (Sub 60°)')):
            ax.set_title(title, fontsize=9, color='#595959')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_ylabel('Luminance (nit)')
        self.ax_sub.set_xlabel('Gray Level')

        # 시리즈 컨테이너
        # key: ('OFF'|'ON', role='main'|'sub', pattern)
        self._lines = {}
        self._data  = {}

        # OFF 시리즈 미리 생성 (실선: ON, 점선: OFF)
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            for pat, col in self._PAT_COLORS.items():
                k_off = ('OFF', role, pat)
                line_off, = ax.plot([], [], linestyle='--', color=col, label=f'OFF {role} {pat}')
                self._lines[k_off] = line_off
                self._data[k_off]  = {'x':[], 'y':[]}

                k_on = ('ON', role, pat)
                line_on, = ax.plot([], [], linestyle='-', color=col, label=f'ON {role} {pat}')
                self._lines[k_on] = line_on
                self._data[k_on]  = {'x':[], 'y':[]}

        self._update_legends()
        self.canvas.draw_idle()

    def reset_on(self):
        """ON 시리즈 전부 리셋 (보정 후 재측정 등 새 런을 시작할 때 호출)"""
        for (state, role, pat), line in self._lines.items():
            if state == 'ON':
                self._data[(state, role, pat)]['x'].clear()
                self._data[(state, role, pat)]['y'].clear()
                line.set_data([], [])

        self._autoscale()
        self._update_legends()
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, pattern: str, gray: int, luminance: float):
        """
        state: 'OFF' or 'ON'
        role : 'main' or 'sub'
        pattern: 'white'|'red'|'green'|'blue'
        """
        key = (state, role, pattern)
        if key not in self._lines:
            return
        self._data[key]['x'].append(int(gray))
        self._data[key]['y'].append(float(luminance))
        self._lines[key].set_data(self._data[key]['x'], self._data[key]['y'])

        # 축 자동 스케일
        self._autoscale(lazy_role=role)
        self.canvas.draw_idle()

    # 내부 유틸
    def _autoscale(self, lazy_role=None):
        roles = [lazy_role] if lazy_role in ('main', 'sub') else ('main', 'sub')
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            if role not in roles: continue
            ys = []
            for (state, r, pat), line in self._lines.items():
                if r==role and line.get_xdata() and line.get_ydata():
                    ys.extend(line.get_ydata())
            if ys:
                ymax = max(ys)
                ax.set_ylim(0, max(1e-6, ymax) * 1.05)

    def _update_legends(self):
        for ax in (self.ax_main, self.ax_sub):
            handles, labels = [], []
            for (state, role, pat), line in self._lines.items():
                if line.get_xdata() and line.get_ydata() and line.axes is ax:
                    handles.append(line); labels.append(line.get_label())
            if handles:
                ax.legend(handles, labels, fontsize=7, loc='upper left')
            else:
                leg = ax.get_legend()
                if leg: leg.remove()
                    
                    
# ─────────────────────────────────────────────────────────────────────────
# CIE1976ChartVAC: OFF는 참조, ON은 매 회차 리셋 (점만 찍음)
# ─────────────────────────────────────────────────────────────────────────
class CIE1976ChartVAC:
    def __init__(self, target_widget, title='Color Shift (u′v′)'):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # 배경/축
        self.ax.set_title(title, fontsize=9, color='#595959')
        self.ax.set_xlim(0.0, 0.70)
        self.ax.set_ylim(0.0, 0.60)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_xlabel("u′"); self.ax.set_ylabel("v′")

        # 참조 경계(옵션) — 필요 없으면 주석
        # ... (BT.709, DCI 등)

        # 점 시리즈 생성
        # key: ('OFF'|'ON', 'main'|'sub')
        self._lines = {}
        self._data  = {}
        styles = {
            ('OFF','main'): dict(marker='o', mfc='none', mec='green', ls='None', ms=4, label='OFF main'),
            ('OFF','sub'):  dict(marker='o', mfc='green', mec='green', ls='None', ms=4, label='OFF sub'),
            ('ON','main'):  dict(marker='s', mfc='none', mec='red',   ls='None', ms=4, label='ON main'),
            ('ON','sub'):   dict(marker='s', mfc='red',  mec='red',   ls='None', ms=4, label='ON sub'),
        }
        for k, st in styles.items():
            ln, = self.ax.plot([], [], **st)
            self._lines[k] = ln
            self._data[k]  = {'u':[], 'v':[]}

        self.ax.legend(loc='lower right', fontsize=7)
        self.canvas.draw_idle()

    def reset_on(self):
        """ON 포인트 전부 리셋"""
        for (state, role), ln in self._lines.items():
            if state == 'ON':
                self._data[(state, role)]['u'].clear()
                self._data[(state, role)]['v'].clear()
                ln.set_data([], [])
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, u_p: float, v_p: float):
        key = (state, role)
        if key not in self._lines: return
        self._data[key]['u'].append(float(u_p))
        self._data[key]['v'].append(float(v_p))
        self._lines[key].set_data(self._data[key]['u'], self._data[key]['v'])
        self.canvas.draw_idle()
        
# ─────────────────────────────────────────────────────────────────────────
# LUTChartVAC: 매번 reset_and_plot(lut_dict) 호출로 새로 그림
# lut_dict: {"R_Low":[4096], "R_High":[4096], ...}
# ─────────────────────────────────────────────────────────────────────────
class LUTChartVAC:
    def __init__(self, target_widget, title='TV LUT (12-bit)'):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        self.ax.set_title(title, fontsize=9, color='#595959')
        self.ax.set_xlim(0, 4095)
        self.ax.set_ylim(0, 4095)
        self.ax.set_xlabel('Gray Level (12-bit)')
        self.ax.set_ylabel('Input Level')
        self.ax.grid(True, linestyle='--', alpha=0.3)

        self._lines = {}  # key: 'R_Low'...'B_High'
        self.canvas.draw_idle()

    def reset_and_plot(self, lut_dict: dict):
        """기존 라인 전부 제거 후 새 LUT로 그리기"""
        # clear
        for ln in self._lines.values():
            try:
                ln.remove()
            except Exception:
                pass
        self._lines.clear()

        xs = np.arange(4096)
        styles = {
            'R_Low':  dict(color='red',   ls='--'),
            'R_High': dict(color='red',   ls='-'),
            'G_Low':  dict(color='green', ls='--'),
            'G_High': dict(color='green', ls='-'),
            'B_Low':  dict(color='blue',  ls='--'),
            'B_High': dict(color='blue',  ls='-'),
        }
        for k, st in styles.items():
            ys = np.asarray(lut_dict[k], dtype=float).ravel()
            ln, = self.ax.plot(xs, ys, **st, label=k.replace('_',' '))
            self._lines[k] = ln

        # 축 재스케일, 레전드
        self.ax.set_xlim(0, 4095)
        ymax = 0.0
        for ln in self._lines.values():
            if ln.get_ydata().size:
                ymax = max(ymax, float(np.nanmax(ln.get_ydata())))
        self.ax.set_ylim(0, max(4095, ymax*1.05))
        self.ax.legend(fontsize=7, loc='upper left')
        self.canvas.draw_idle()
        

