1. vac_table_chromaticityDiff라는 qtablewidget에는 제목행/열을 제외한 두번째 열에서 두번째 행에는 델타 Cx "spec 만족 수/전체 수" 표기,  세번째 행에는 델타 Cy "spec 만족 수/전체 수" 표기, 네번째 행에는 델타 감마 "spec 만족 수/전체 수" 표기 
2. vac_chart_chromaticityDiff라는 qvboxlayout에는 gray scale에 따른 VAC OFF와 VAC ON(보정후)에서의 Cx, Cy 값 업데이트
3. vac_table_gammaLinearity라는 qtablewidget에는 제목행/열을 제외한 두번째 열의 두번째 행에 VAC OFF에서의 88~232gray scale을 8씩 증가하면서 각 구간별 기울기의 평균, 세번째 열의 두번째 행에 VAC ON에서의 88~232gray scale을 8씩 증가하면서 각 구간별 기울기의 평균
4. vac_chart_gammaLinearity라는 qvboxlayout에는 VAC OFF에서의 88~232gray scale을 8씩 증가하면서 각 구간별 기울기값과 VAC ON에서의 88~232gray scale을 8씩 증가하면서 각 구간별 기울기값 (dot+line) (88-96 사이 기울기면 x축 88 gray와 96 gray 사이 에 기울기 값 좌표 위치하는 식으로 차트 그려주세요)
5. vac_table_colorShift_3라는 qtablewidget에는 제목행/열을 제외한 두번째 열은 VAC OFF 데이터를 넣을 거고 이 열에서 두번째 행은 DarkSkin 패턴의 정면-측면 u`v` 빼기제곱루트값, 세번째 행은 LightSkin 패턴의 정면-측면 u`v` 빼기제곱루트값, 네번째 행은 Asian 패턴의 정면-측면 u`v` 빼기제곱루트값, 다섯번째 행은 Western 패턴의 정면-측면 u`v` 빼기제곱루트값, 여섯번째 행은 위 패턴에서의 정면-측면 u`v` 빼기제곱루트값들의 평균
세번째 열은 VAC ON 데이터고 이 열에서 두번째 행은 DarkSkin 패턴의 정면-측면 u`v` 빼기제곱루트값, 세번째 행은 LightSkin 패턴의 정면-측면 u`v` 빼기제곱루트값, 네번째 행은 Asian 패턴의 정면-측면 u`v` 빼기제곱루트값, 다섯번째 행은 Western 패턴의 정면-측면 u`v` 빼기제곱루트값, 여섯번째 행은 위 패턴에서의 정면-측면 u`v` 빼기제곱루트값들의 평균
6. vac_chart_colorShift_3라는 qvboxlayout에는 barchart를 넣을 건데 각 패턴당 (VAC OFF,VAC ON)이렇게 묶어서 묶음이 패턴이 4개 있으니까 4개 있으면 됩니다.

참고로 그래프 초기화 클래스는 아래와 같습니다.
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtWidgets import QSizePolicy

import modules.chart_style as cs  # 스타일 함수 모듈

class BarChart:
    def __init__(self, target_widget, title='Bar Chart',
                 x_labels=None, y_label='Y Axis',
                 y_range=(0, 0.06), y_tick=0.01, spec_line=0.04,
                 bar_color='gray'):

        # Default x labels
        if x_labels is None:
            x_labels = ['A', 'B', 'C']

        self.x_labels = x_labels
        self.y_label = y_label
        self.y_range = y_range
        self.y_tick = y_tick
        self.spec_line = spec_line
        self.bar_color = bar_color

        # Create chart
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        self._init_style(title)
        self._init_chart()

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.draw()

    def _init_style(self, title):
        cs.MatFormat_ChartArea(self.fig, left=0.1, right=0.95, top=0.9, bottom=0.1)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title)
        cs.MatFormat_AxisTitle(self.ax, self.y_label, axis='y')
        cs.MatFormat_Axis(self.ax, self.y_range[0], self.y_range[1], self.y_tick, axis='y')
        self.ax.set_xticks(np.arange(len(self.x_labels)))
        self.ax.set_xticklabels(self.x_labels, fontsize=9, color='#595959')
        self.ax.tick_params(axis='x', length=0)

    def _init_chart(self, data=None):
        if data is None:
            data = [0] * len(self.x_labels)

        self.ax.clear()
        self._init_style(self.ax.get_title())  # Re-apply style
        self.ax.bar(np.arange(len(self.x_labels)), data, color=self.bar_color)
        self.ax.axhline(y=self.spec_line, color='red', linestyle='--', linewidth=0.8)

    def update_data(self, data):
        """Update bar chart with new data"""
        self._init_chart(data)
        self.canvas.draw()

최적화 루프에서 아래와 같이 로딩했고

        self.vac_optimization_chromaticity_chart = XYChart(
            target_widget=self.ui.vac_chart_chromaticityDiff,
            x_label='Gray Level',
            y_label='Cx/Cy',
            x_range=(0, 256),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title=None,
            title_color='#595959',
            legend=False
        )
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
        self.vac_optimization_colorshift_chart = BarChart(
            target_widget=self.ui.vac_chart_colorShift_3,
            title='Skin Color Shift',
            x_labels=self.colorshift_x_labels,
            y_label='delta u`v`',
            spec_line=0.04

만약 제가 요구한 기능에 맞게 클래스 수정이 필요하다면 수정해주셔도 돼요.

또 참고로 op.colorshift_patterns는 아래와 같아요:
colorshift_patterns = [
    ("Red", 177, 51, 60),
    ("Green", 71, 149, 71),
    ("Blue", 49, 64, 150),
    ("Cyan", 0, 136, 165),
    ("Magenta", 187, 85, 147),
    ("Yellow", 237, 199, 32),
    ("White", 255, 255, 255),
    ("Gray", 128, 128, 128),
    ("Dark Skin", 116, 80, 66),
    ("Light Skin", 196, 150, 129),
    ("Asian", 196, 147, 118),
    ("Western", 183, 130, 93)
]
