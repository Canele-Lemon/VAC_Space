class XYChart:
    ...

    def set_series(self, key, x_list, y_list, *, marker=None, linestyle='-', label=None, axis_index=0, autoscale=True):
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

        # 3) 데이터 반영
        self.data[key]['x'] = x_arr.tolist()
        self.data[key]['y'] = y_arr.tolist()

        ln = self.lines[key]
        if marker is not None:
            ln.set_marker(marker)
        ln.set_linestyle(linestyle)
        if label is not None:
            ln.set_label(label)
        ln.set_data(self.data[key]['x'], self.data[key]['y'])

        ax = ln.axes
        # ★ 필요할 때만 autoscale
        if autoscale:
            ax.relim()
            ax.autoscale_view()

        self.update_legend()
        self.canvas.draw()
        
class XYChart:
    ...

    def set_y_axis_range(self, min_val, max_val, tick_count=None, axis_index=0):
        """
        y축 범위 및 tick 개수 설정.
        tick_count가 주어지면 (max-min)/(tick_count-1) 간격으로 tick 생성.
        """
        if axis_index >= len(self.axes):
            print(f"[XYChart] Invalid axis index in set_y_axis_range: {axis_index}")
            return

        ax = self.axes[axis_index]

        if tick_count is not None and tick_count > 1:
            tick_interval = float(max_val - min_val) / float(tick_count - 1)
        else:
            tick_interval = None

        cs.MatFormat_Axis(
            ax,
            min_val=float(min_val),
            max_val=float(max_val),
            tick_interval=tick_interval,
            axis='y'
        )
        # x축은 건드리지 않고, y축은 우리가 지정한 범위 유지
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=False)
        self.canvas.draw()
        
        .