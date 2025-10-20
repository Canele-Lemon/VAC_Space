1. self.vac_optimization_gamma_chart: main 계측기 (inst1)과 sub 계측기 (inst2)로 측정한 정면/측면의 gray level에 따른 휘도 데이터가 업데이트 되는 차트입니다.(위/아래) VAC OFF 데이터는 항상 ref로 남아있으며 VAC ON, 즉 lut 보정 후 측정 데이터는 기존 VAC ON line을 reset하고 새로 업데이트됩니다.
2. self.vac_optimization_cie1976_chart: main 계측기 (inst1)과 sub 계측기 (inst2)로 측정한 정면/측면의 맥베스 패턴에 따른 uv prime 좌표가 업데이트 되는 차트입니다. VAC OFF 데이터는 항상 ref로 남아있으며 VAC ON, 즉 lut 보정 후 측정 데이터는 기존 VAC ON point를 reset하고 새로 업데이트됩니다.
3. self.vac_optimization_lut_chart: tv에 writing되는 lut 차트입니다. 처음 db fetch를 통해 받아온 lut를 그릴 때, lut 보정 후 보정된 lut를 보여줄 때마다 원래 그려져 있던 line이 reset되고 새로 업데이트됩니다.
4. self.vac_optimization_chromaticity_chart
5.
6.

class Widget_vacspace(QWidget):
    def __init__(self, parent=None):
        super(Widget_vacspace, self).__init__(parent)
        self.ui = Ui_vacspaceForm()
        self.ui.setupUi(self)
        
        self.ui.vac_btn_startOptimization.clicked.connect(self.start_VAC_optimization)

        self._vac_dict_cache = None
        
        self.vac_optimization_gamma_chart = GammaChart(
            target_widget=self.ui.vac_chart_gamma_3,
            multi_axes=True,
            num_axes=2
        )
        
        self.vac_optimization_cie1976_chart = CIE1976ChromaticityDiagram(self.ui.vac_chart_colorShift_2, 
                                                                            "Color Shift",
                                                                            left_margin=0.12, 
                                                                            right_margin=0.95, 
                                                                            top_margin=0.90, 
                                                                            bottom_margin=0.15)

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
        )
