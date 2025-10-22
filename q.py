import numpy as np

class XYChart:
    ...
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