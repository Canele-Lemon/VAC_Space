1.

class GammaChart:
    def __init__(self, target_widget, multi_axes=False, num_axes=1):
        # XYChart 인스턴스 생성
        self.chart = XYChart(
            target_widget=target_widget,
            x_label='Gray Level',
            y_label='Luminance (nit)',
            x_range=(0, 255),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title='Gamma Measurement',
            multi_axes=multi_axes,
            num_axes=num_axes,
            layout='vertical',
            share_x=True
        )
        self._init_lines()

    def _init_lines(self):
        # 측정 조건별 선 추가
        colors = {
            'W': 'gray',
            'R': 'red',
            'G': 'green',
            'B': 'blue'
        }

        # 기본적으로 첫 번째 축에 선 추가
        for angle in [0, 60]:
            for data_label in ['data_1', 'data_2']:
                for color_key, color_val in colors.items():
                    key = f'{angle}deg_{color_key}_{data_label}'
                    axis_index = 0  # 첫 번째 축
                    self.chart.add_line(key, color=color_val, linestyle='--' if angle == 0 else '-', label=key, axis_index=axis_index)

        # DQA용 선은 두 번째 축에 추가 (있다면)
        for data_label in ['data_1', 'data_2']:
            key = f'60deg_dqa_{data_label}'
            dot_color = 'lightgray' if data_label == 'data_1' else 'darkgray'
            axis_index = 1 if len(self.chart.axes) > 1 else 0
            self.chart.add_line(key, color=dot_color, marker='*', linestyle='None', label=key, axis_index=axis_index)

    def add_series(self, axis_index=0, label=None, color=None, linestyle='-'):
        key = f"{label or 'series'}_{axis_index}_{len(self.chart.lines)}"
        self.chart.add_line(key, color=color or 'black', linestyle=linestyle, label=label, axis_index=axis_index)
        return self.chart.lines[key]

    def autoscale(self):
        # XYChart가 relim/autoscale_view를 update에서 하긴 하지만, 외부에서 강제 호출용
        for ax in self.chart.axes:
            ax.relim(); ax.autoscale_view()
        self.canvas.draw_idle() if hasattr(self, 'canvas') else self.chart.canvas.draw_idle()

    def draw(self):
        self.chart.canvas.draw_idle()
        
    def update_from_measurement(self, color, lv, viewangle, data_label, vac_status):
        try:
            lv = float(lv)
        except ValueError:
            print(f"[GammaChart] Invalid luminance value: {lv}")
            return

        if color == 'DQA':
            key = f'60deg_dqa_{data_label}'
            x_data = [0, 128, 200, 255][:len(self.chart.data[key]['y']) + 1]
        else:
            key = f'{viewangle}deg_{color}_{data_label}'
            from modules import op  # gray_levels 사용
            x_data = op.gray_levels[:len(self.chart.data[key]['y']) + 1]

        if key in self.chart.data and len(x_data) == len(self.chart.data[key]['y']) + 1:
            self.chart.update(key, x_data[-1], lv)
            label = f'Data #{data_label[-1]} {viewangle}° {"(DQA) " if color == "DQA" else ""}{vac_status}'
            self.chart.set_label(key, label)

CammaChart 클래스에는 다음과 같이 add_series가 있는데 잘 작성되었나요?


2. op.gray_levels_256은 정의되어있으니 걱정하지 않으셔도 됩니다.

3. build_vacparam_std_format을 통해 실제 tv에 적용되는 json 포맷으로 변환되는게 맞겠지요,,? 실제 json 포멧에는 다 tab으로 문자 밎 숫자가 띄어쓰기 되어 있어요.
