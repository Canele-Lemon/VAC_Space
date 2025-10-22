2025-10-23 07:56:22,801 - INFO - subpage_vacspace.py:908 - [LUT LOADING] DB fetch LUT TV Writing 확인을 위한 TV Reading 완료
2025-10-23 07:56:23,127 - INFO - subpage_vacspace.py:940 - [MES] DB fetch LUT 기준 측정 시작
2025-10-23 07:59:42,662 - INFO - subpage_vacspace.py:1111 - [SPEC] max|ΔGamma|=0.578095 (≤0.05), max|ΔCx|=0.006300, max|ΔCy|=0.024400 (≤0.003)
2025-10-23 07:59:42,664 - ERROR - subpage_vacspace.py:1422 - 'list' object has no attribute 'size'
Traceback (most recent call last):
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1420, in _finalize_session
    s['on_done'](s['store'])
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 933, in _after_on
    self._update_spec_views(self._off_store, self._on_store)  # ← 여기!
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1648, in _update_spec_views
    self.vac_optimization_chromaticity_chart.set_series(
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\charts\xy_chart.py", line 111, in set_series
    self.update_legend()
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\charts\xy_chart.py", line 87, in update_legend
    filtered = [(h, l) for h, l in zip(handles, labels) if h.get_xdata().size > 0 and h.get_ydata().size > 0]
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\charts\xy_chart.py", line 87, in <listcomp>
    filtered = [(h, l) for h, l in zip(handles, labels) if h.get_xdata().size > 0 and h.get_ydata().size > 0]
AttributeError: 'list' object has no attribute 'size'

이런에러가 발생했습니다

참고로 XYChart 클래스:
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
            # Filter out lines with no data
            filtered = [(h, l) for h, l in zip(handles, labels) if h.get_xdata().size > 0 and h.get_ydata().size > 0]
            if filtered:
                handles, labels = zip(*filtered)
                ax.legend(handles, labels, loc='upper right', fontsize=8)
            else:
                ax.legend().remove()
                
    def set_series(self, key, x_list, y_list, *, marker=None, linestyle='-', label=None, axis_index=0):
        """한 번에 x/y 전체를 세팅 (streaming 말고 배치 갱신)"""
        if key not in self.lines:
            # 미리 add_line 되어있지 않으면 생성
            axis = self.axes[axis_index]
            line, = axis.plot([], [], linestyle=linestyle, marker=marker, label=label or key)
            self.lines[key] = line
            self.data[key] = {'x': [], 'y': []}
        self.data[key]['x'] = list(x_list)
        self.data[key]['y'] = list(y_list)
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



