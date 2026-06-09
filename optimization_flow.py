def _update_legends(self):
    for ax in (self.ax_main, self.ax_sub):
        handles, labels = [], []

        for (state, role, pat), ln in self._lines.items():
            x = ln.get_xdata()
            y = ln.get_ydata()

            if ln.axes is ax and len(x) > 0 and len(y) > 0:
                handles.append(ln)
                labels.append(ln.get_label())

        if handles:
            ax.legend(handles, labels, fontsize=8, loc='upper left')
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()