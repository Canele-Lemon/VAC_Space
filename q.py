    def initialize_measurement_colorshift_chart(self):
        self.fig_cie1976, self.ax_cie1976 = plt.subplots()
        self.canvas_cie1976 = FigureCanvas(self.fig_cie1976)
        self.ui.vac_chart_colorShift.addWidget(self.canvas_cie1976)

        BT709_u, BT709_v = cf.convert2DlistToPlot(op.BT709_uvprime)
        DCI_u, DCI_v = cf.convert2DlistToPlot(op.DCI_uvprime)
        CIE1976_u = [item[1] for item in op.CIE1976_uvprime]
        CIE1976_v = [item[2] for item in op.CIE1976_uvprime]

        self.fig_cie1976.clear()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = cf.get_normalized_path(__file__, '..','..', 'resources/images/pictures', 'cie1976 (2).png')
        img_cie1976 = plt.imread(image_path, format='png')
        
        self.ax_cie1976 = self.fig_cie1976.add_subplot(111)
        self.ax_cie1976.imshow(img_cie1976, extent=[0, 0.70, 0, 0.60])
        
        cs.MatFormat_ChartArea(self.fig_cie1976, left=0.10, right=0.95, top=0.95, bottom=0.10)
        cs.MatFormat_FigArea(self.ax_cie1976)
        cs.MatFormat_AxisTitle(self.ax_cie1976, axis_title='u`', axis='x')
        cs.MatFormat_AxisTitle(self.ax_cie1976, axis_title='v`', axis='y')
        cs.MatFormat_Axis(self.ax_cie1976, min_val=0, max_val=0.7, tick_interval=0.1, axis='x')
        cs.MatFormat_Axis(self.ax_cie1976, min_val=0, max_val=0.6, tick_interval=0.1, axis='y')
        cs.MatFormat_Gridline(self.ax_cie1976, linestyle='--')
        
        self.ax_cie1976.plot(BT709_u, BT709_v, color='black', linestyle='--', linewidth=0.8, label="BT.709")
        self.ax_cie1976.plot(DCI_u, DCI_v, color='black', linestyle='-', linewidth=0.8, label="DCI")
        self.ax_cie1976.plot(CIE1976_u, CIE1976_v, color='black', linestyle='-', linewidth=0.3, label=None)
        self.line_cie1976_data_1_0deg = self.ax_cie1976.plot([], [], 'o', markerfacecolor='none', markeredgecolor='red')[0]
        self.line_cie1976_data_1_60deg = self.ax_cie1976.plot([], [], 's', markerfacecolor='none', markeredgecolor='red')[0]
        self.line_cie1976_data_2_0deg = self.ax_cie1976.plot([], [], 'o', markerfacecolor='none', markeredgecolor='green')[0]
        self.line_cie1976_data_2_60deg = self.ax_cie1976.plot([], [], 's', markerfacecolor='none', markeredgecolor='green')[0]
        self.ax_cie1976.legend(fontsize=9)

        self.canvas_cie1976.draw()
