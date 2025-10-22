class CIE1976Chart:
    def __init__(self, parent=None):
        ...
        ms = 4.0  # 마커 크기 조금 작게
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

        # 🔹 데이터 저장용 딕셔너리
        self.data = {k: {'u': [], 'v': []} for k in self.lines.keys()}  # (self._data → self.data)
        self._update_legend()

    def reset_on(self):
        for k in (('ON', 'main'), ('ON', 'sub')):
            self.data[k]['u'].clear()
            self.data[k]['v'].clear()
            self.lines[k].set_data([], [])
        self._update_legend()
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, u_p: float, v_p: float):
        key = (state, role)
        if key not in self.lines:
            return
        self.data[key]['u'].append(float(u_p))
        self.data[key]['v'].append(float(v_p))
        self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
        self.lines[key].set_label(f"{state} {role}")
        self._update_legend()
        self.canvas.draw_idle()

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