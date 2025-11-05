        self.vac_optimization_gammalinearity_chart = XYChart(
            target_widget=self.ui.vac_chart_gammaLinearity,
            x_label='Gray Level',
            y_label='Slope',
            x_range=(0, 256),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title=None,
            title_color='#595959',
            legend=False
        )
의 클래스인 아래 XYChart가 tick 관리하도록 수정해도 괜찮나요?



import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import modules.chart_style as cs
import numpy as np

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
