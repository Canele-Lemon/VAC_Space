    def _update_lut_chart_and_table(self, lut_dict):
        """
        self.vac_optimization_lut_chart (x:0~4095) 갱신 + self.ui.vac_table_rbgLUT_4에 숫자 뿌리기
        이미 사용중인 update_rgbchannel_chart/update_rgbchannel_table 재사용해도 됩니다.
        """
        try:
            df = pd.DataFrame({
                "R_Low":  lut_dict["R_Low"],  "R_High": lut_dict["R_High"],
                "G_Low":  lut_dict["G_Low"],  "G_High": lut_dict["G_High"],
                "B_Low":  lut_dict["B_Low"],  "B_High": lut_dict["B_High"],
            })
            # 예: 기존 메서드 재사용
            self.update_rgbchannel_chart(
                df,
                self.graph['vac_laboratory']['data_acquisition_system']['input']['ax'],
                self.graph['vac_laboratory']['data_acquisition_system']['input']['canvas']
            )
            self.update_rgbchannel_table(df, self.ui.vac_table_rbgLUT_4)
        except Exception as e:
            logging.exception(e)

에서 업데이트할 그래프는 다음과 같이 초기화했어요:

        self.vac_optimization_lut_chart = XYChart(
            target_widget=self.ui.vac_graph_rgbLUT_4,
            x_label='Gray Level (12-bit)',
            y_label='Input Level',
            x_range=(0, 4095),
            y_range=(0, 4095),
            x_tick=512,
            y_tick=512,
            title=None,
            title_color='#595959',
            legend=False
        )

참고로 XYChart 클래스는 다음과 같아요
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
            
            # X축 라벨은 마지막 축에만 표시
            if i == len(self.axes) - 1:
                cs.MatFormat_AxisTitle(ax, axis_title=x_label, axis='x')
            else:

                ax.set_xticklabels([])  # 상단 축의 X축 눈금 라벨 제거
                ax.set_xlabel('')       # X축 제목 제거
                

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

_update_lut_chart_and_table 메서드를 어떻게 수정하면 될까요?
