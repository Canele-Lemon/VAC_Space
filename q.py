class Widget_vacspace(QWidget):
    def __init__(self, parent=None):
        self.ui.vac_btn_startOptimization.clicked.connect(self.start_VAC_optimization)
        
        self.vac_optimization_gamma_chart = GammaChart(
            target_widget=self.ui.vac_chart_gamma_3,
            multi_axes=True,
            num_axes=2
        )

        self.vac_optimization_colorshift_chart = CIE1976ChromaticityDiagram(self.ui.vac_chart_colorShift_2)

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


이런식으로 이미 각 차트를 ready 해 놓았습니다. self.vac_optimization_colorshift_chart는 color shift 측정 결과를 업데이트 하면 됩니다. 참고로 CIE1976ChromaticityDiagram 클래스는 아래와 같습니다
class CIE1976ChromaticityDiagram:
    def __init__(self, target_widget):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        self._init_background()
        self._init_reference_lines()
        self._init_data_lines()
        self._init_data_storage()

        self.canvas.draw()

    def _init_background(self):
        image_path = cf.get_normalized_path(__file__, '..', '..', '..', 'resources/images/pictures', 'cie1976 (2).png')
        img = plt.imread(image_path, format='png')
        self.ax.imshow(img, extent=[0, 0.70, 0, 0.60])

        cs.MatFormat_ChartArea(self.fig, left=0.10, right=0.95, top=0.95, bottom=0.10)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_AxisTitle(self.ax, axis_title='u`', axis='x')
        cs.MatFormat_AxisTitle(self.ax, axis_title='v`', axis='y')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.7, tick_interval=0.1, axis='x')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.6, tick_interval=0.1, axis='y')
        cs.MatFormat_Gridline(self.ax, linestyle='--')

    def _init_reference_lines(self):
        BT709_u, BT709_v = cf.convert2DlistToPlot(op.BT709_uvprime)
        DCI_u, DCI_v = cf.convert2DlistToPlot(op.DCI_uvprime)
        CIE1976_u = [item[1] for item in op.CIE1976_uvprime]
        CIE1976_v = [item[2] for item in op.CIE1976_uvprime]

        self.ax.plot(BT709_u, BT709_v, color='black', linestyle='--', linewidth=0.8, label="BT.709")
        self.ax.plot(DCI_u, DCI_v, color='black', linestyle='-', linewidth=0.8, label="DCI")
        self.ax.plot(CIE1976_u, CIE1976_v, color='black', linestyle='-', linewidth=0.3)

    def _init_data_lines(self):
        self.lines = {
            'data_1_0deg': self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='red')[0],
            'data_1_60deg': self.ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='red')[0],
            'data_2_0deg': self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='green')[0],
            'data_2_60deg': self.ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='green')[0],
        }

    def _init_data_storage(self):
        self.data = {
            'data_1_0deg': {'u': [], 'v': []},
            'data_1_60deg': {'u': [], 'v': []},
            'data_2_0deg': {'u': [], 'v': []},
            'data_2_60deg': {'u': [], 'v': []},
        }

    def update(self, u_p, v_p, data_label, view_angle, vac_status):
        key = f'{data_label}_{view_angle}deg'
        self.data[key]['u'].append(float(u_p))
        self.data[key]['v'].append(float(v_p))

        self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
        self.lines[key].set_label(f'Data #{data_label[-1]} {view_angle}° {vac_status}')

        self.ax.legend(fontsize=9)
        self.canvas.draw()

이제 다시 Widget_vacspace 클래스에 
    def start_VAC_optimization(self):
        """
        =============== 메인 엔트리: 버튼 이벤트 연결용 ===============
        전체 플로우:
        1) VAC OFF 보장 → 측정(OFF baseline) + UI 업데이트
        2) DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 적용(ON) → 측정(ON 현재) + UI 업데이트
        3) 스펙 확인 → 통과면 종료
        4) 미통과면 자코비안 기반 보정(256기준) → 4096 보간 반영 → TV 적용 → 재측정 → 스펙 재확인
        5) (필요 시 반복 2~3회만)
        """
        try:
            # (0) 자코비안 로드
            self.model_dir = r"D:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\model"
            jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_Y0_high.pkl')
            logging.debug(f"Jacobian path: {jac_path}")
            artifacts = joblib.load(jac_path)
            
            print("======================= artifacts 구조 확인 =======================")
            logging.debug(f"Artifacts keys: {artifacts.keys()}")
            logging.debug(f"Components: {artifacts['components'].keys()}")

            A_Gamma = self.build_A_from_artifacts(artifacts, "Gamma")  # (256, 3K)
            A_Cx    = self.build_A_from_artifacts(artifacts, "Cx")
            A_Cy    = self.build_A_from_artifacts(artifacts, "Cy")
            
            print("======================= A 행렬 shape 확인 =======================")
            logging.debug(f"A_Gamma shape:": {A_Gamma.shape})
            logging.debug(f"A_Cx shape:" {A_Cx.shape})
            logging.debug(f"A_Cy shape:" {A_Cy.shape})


            # (1) VAC OFF 보장 + 측정
            self.ui.vac_btn_startOptimization.setEnabled(False)
            self.start_viewing_angle_char_measurement()

start_VAC_optimization메서드에서 start_viewing_angle_char_measurement 부분을 어떻게 작성해야 할 지 알려주세요.
현재까지 작성한 건 아래와 같아요:
    def start_viewing_angle_char_measurement(self):
        # 멀티축 감마 차트 준비(위=main, 아래=sub)
        self.lines_gamma = {
            'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"OFF-main-{p}") for p in ['white','red','green','blue']},
            'sub' : {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=f"OFF-sub-{p}")  for p in ['white','red','green','blue']},
        }

        # 컬러쉬프트 차트 준비(둘이 같은 축)
        fig, ax, canvas = _mk_canvas_in_layout(self, self.ui.vac_chart_colorShift_2)
        ax.set_title("Color Shift (xy)"); ax.set_xlabel("x"); ax.set_ylabel("y")
        main_line, = ax.plot([], [], marker='o', linestyle='None', label='OFF-main')
        sub_line,  = ax.plot([], [], marker='^', linestyle='None', label='OFF-sub')
        ax.legend(); canvas.draw()
        self.cs_ax, self.cs_canvas = ax, canvas
        self.lines_cs = {'main': main_line, 'sub': sub_line}

        # OFF 결과 저장용 버퍼
        self.off_records = {
            'gamma': {'main': {}, 'sub': {}},   # e.g., {'white': {gray: (lv, cx, cy), ...}}
            'colorshift': {'main': [], 'sub': []}  # list of (x, y)
        }

        # 루프 상태
        self._meas_patterns = ['white','red','green','blue']
        self._meas_gray = list(range(256))
        self._pi = 0; self._gi = 0

        # 시작
        self._run_one_step_off()

가능하면 data collection 할 때 사용한 measurement 관련 함수와 비슷하게 작성할 수 있을까요?

참고-data collection  로직
    ##-- Data Collection --##
    def start_data_collection(self):
        self.loading_label_dbconn, self.loading_movie_dbconn = self.start_loading_animation(self.ui.vac_btn_Start, 'loading_spinner.gif')
        
        pk_start = int(self.ui.vac_lineEdit_startPK.text())
        pk_end = int(self.ui.vac_lineEdit_endPK.text())
        exclude_text = self.ui.vac_lineEdit_excludePKs.text()
        exclude_list = [int(pk.strip()) for pk in exclude_text.split(',') if pk.strip().isdigit()]
        
        self.vac_fetch_thread = VACDataFetchFromDBThread(pk_start, pk_end, exclude_list)
        self.vac_fetch_thread.data_fetched.connect(self.on_vac_data_fetched)
        self.vac_fetch_thread.start()
        
    def on_vac_data_fetched(self, filtered_vacdata_list):
        self.stop_loading_animation(self.loading_label_dbconn, self.loading_movie_dbconn)
        self.ui.vac_btn_Start.setVisible(False)
        self.ui.vac_btn_Pause.setVisible(True)
        self.ui.vac_btn_Stop.setVisible(True)

        # self.vacdata_queue = filtered_vacdata_list[:1]
        self.vacdata_queue = filtered_vacdata_list
        self.total_loop_count = len(self.vacdata_queue)
        self.completed_count = 0
        
        self.pk_to_row_index = {}

        self.ui.vac_progressBar_dataCol.setMinimum(0)
        self.ui.vac_progressBar_dataCol.setMaximum(self.total_loop_count)
        self.ui.vac_progressBar_dataCol.setValue(0)
        
        self.ui.vac_tableWidget_dataCollectionLog.setRowCount(self.total_loop_count)
        for index, vacdata in enumerate(self.vacdata_queue):
            pk, vac_version, vac_data = vacdata
            self.pk_to_row_index[pk] = index
            self.append_data_collection_log(pk, vac_version, "Pending")
        
        self.process_next_vacdata()
        
    def process_next_vacdata(self):
        if not self.vacdata_queue:
            logging.info("[Data Collection Complete] Finished processing all VAC data entries.")
            self.ui.vac_btn_Start.setVisible(True)
            self.ui.vac_btn_Pause.setVisible(False)
            self.ui.vac_btn_Stop.setVisible(False)
            return
        
        self.ui.vac_label_totalVACDataCount.setText(
            f"{self.completed_count + 1} / {self.total_loop_count}"
        )

        pk, vac_version, vac_data = self.vacdata_queue.pop(0) # pop(0): 첫 번째 요소 꺼낸 후 리스트에서 제거
        logging.info(f"[Data Collection - Writing] Initiating VAC data writing step: PK = {pk}, VAC Version = {vac_version}")
        self.append_data_collection_log(pk, vac_version, "Processing")
        self.loading_label_datacol, self.loading_movie_datacol = self.start_loading_animation(self.ui.vac_graph_rgbLUT_3, 'loading_Pie.gif', 20)
        self.write_random_VAC_thread = WriteVACdataThread(
            parent=self,
            ser_tv=self.ser_tv,
            vacdataName=self.vacdataName,
            vacdata_loaded=vac_data
        )
        self.write_random_VAC_thread.write_finished.connect(
            lambda write_success, msg: self.read_random_VAC_data(write_success, msg, pk, vac_version)
        )
        self.write_random_VAC_thread.start()

    def read_random_VAC_data(self, write_success, msg, pk, vac_version):
        if not write_success:
            logging.warning(f"[Data Collection - Writing] VAC data writing failed: {msg}")
            completed_time = datetime.now()
            self.append_data_collection_log(pk, vac_version, "Failed", completed_time)
            self.stop_loading_animation(self.loading_label_datacol, self.loading_movie_datacol)
            self.process_next_vacdata()
            return
        
        logging.info(f"[Data Collection - Writing] VAC data writing succeeded: {msg}")
        logging.info(f"[Data Collection - Reading] VAC data reading step: PK = {pk}, VAC Version = {vac_version}")
        self.read_random_VAC_thread = ReadVACdataThread(
            parent=self,
            ser_tv=self.ser_tv,
            vacdataName=self.vacdataName
        )
        self.read_random_VAC_thread.data_read.connect(
            lambda vac_data: self.process_read_vac_data(vac_data, pk, vac_version)
            )
        self.read_random_VAC_thread.error_occurred.connect(
            lambda msg: self.on_vac_read_error(msg, pk, vac_version)
            )
        self.read_random_VAC_thread.start()

    def process_read_vac_data(self, vac_data, pk, vac_version):
        try:
            self.stop_loading_animation(self.loading_label_datacol, self.loading_movie_datacol)
            
            channels = ['R_Low', 'R_High', 'G_Low', 'G_High', 'B_Low', 'B_High']
            rgb_channel_data = {channel: vac_data.get(channel.replace("_", "channel"), []) for channel in channels}
            rgb_channel_df = pd.DataFrame(rgb_channel_data)
            self.update_rgbchannel_chart(rgb_channel_df, 
                                            self.graph['vac_laboratory']['data_acquisition_system']['input']['ax'],
                                            self.graph['vac_laboratory']['data_acquisition_system']['input']['canvas'])
            self.update_rgbchannel_table(rgb_channel_df, self.ui.vac_table_rbgLUT_3)
            self.send_command(self.ser_tv, 'exit')
        
        except Exception as e:
            logging.error(f"Error processing VAC data: {e}")
            completed_time = datetime.now()
            self.append_data_collection_log(pk, vac_version, "Failed", completed_time)
            self.stop_loading_animation(self.loading_label_datacol, self.loading_movie_datacol)
            self.send_command(self.ser_tv, 'exit')
            return
        
        try:
            self.start_measurement_step(pk, vac_version)

        except Exception as e:
                logging.error(f"[Data Collection - Measurement] Measurement error: PK = {pk}, VAC Version = {vac_version}, Error = {e}")
                completed_time = datetime.now()
                self.append_data_collection_log(pk, vac_version, "Failed", completed_time)
                self.stop_loading_animation(self.loading_label_datacol, self.loading_movie_datacol)
                self.send_command(self.ser_tv, 'exit')
        
    def on_vac_read_error(self, msg, pk, vac_version):
        logging.error(f"[Data Collection - Reading] VAC data reading failed: PK = {pk}, VAC Version = {vac_version}, Error = {msg}")
        self.send_command(self.ser_tv, 'exit')
        self.stop_loading_animation(self.loading_label_datacol, self.loading_movie_datacol)
        self.process_next_vacdata()
        
    def start_measurement_step(self, pk, vac_version):
        logging.info(f"[Data Collection - Measurement] Measurement step: PK = {pk}, VAC Version = {vac_version}")
        
        self.is_paused = False
        self.is_stopped = False
        self.is_running = False
        
        self.reset_output_graph()
        self.mes_Category = "Gamma"
        self.gamma_patterns = ['white', 'red', 'green', 'blue']
        self.gamma_pattern_index = 0
        self.gray_levels = list(range(0, 256))
        # self.gray_levels = op.gray_levels
        self.gray_level_index = 0
        
        self.colorshift_patterns = op.colorshift_patterns
        self.colorshift_patterns_index = 0
          
        self.table_widget_main = self.ui.vac_table_measure_results_main
        self.table_widget_sub = self.ui.vac_table_measure_results_sub
        self.table_widget_main.setRowCount(0)
        self.table_widget_sub.setRowCount(0)

        self.row_main = self.table_widget_main.rowCount()
        self.row_sub = self.table_widget_sub.rowCount()

        self.measurement_finished_callback = lambda: self.on_measurement_finished(pk, vac_version)
        
        self.run_measurement_step()

    def stop_measurement(self):
        logging.info("Measurement stop requested")
        self.ui.vac_btn_Start.setVisible(True)
        self.ui.vac_btn_Pause.setVisible(False)
        self.ui.vac_btn_Resume.setVisible(False)
        self.ui.vac_btn_Stop.setVisible(False)
        
        self.is_stopped = True
        self.is_paused = False

        if hasattr(self, 'main_measure_thread') and self.main_measure_thread.isRunning():
            self.main_measure_thread.cancel()

        if hasattr(self, 'sub_measure_thread') and self.sub_measure_thread.isRunning():
            self.sub_measure_thread.cancel()

    def pause_measurement(self):
        logging.info("Measurement pause requested")
        self.ui.vac_btn_Pause.setVisible(False)
        self.ui.vac_btn_Resume.setVisible(True)  
              
        self.is_paused = True

    def resume_measurement(self):
        logging.info("Measurement resume requested")
        self.ui.vac_btn_Resume.setVisible(False)
        self.ui.vac_btn_Pause.setVisible(True)

        self.is_paused = False
        if not self.is_running:
            self.run_measurement_step()

    def on_measurement_finished(self, pk, vac_version):
        self.upload_collected_data_to_DB(pk, vac_version)
        
        self.completed_count += 1
        self.ui.vac_progressBar_dataCol.setValue(self.completed_count)
        completed_time = datetime.now()
        self.append_data_collection_log(pk, vac_version, "Completed", completed_time)
        logging.info(f"[Data Collection - Measurement] Measurement finished: PK = {pk}, VAC Version = {vac_version}")

        self.process_next_vacdata()
                
    def run_measurement_step(self):
        if self.is_stopped:
            logging.info("Measurement has been stopped. Ending measurement")
            self.is_running = False
            return

        if self.is_paused:
            logging.info("Measurement is paused")
            QTimer.singleShot(500, self.run_measurement_step)
            return

        self.is_running = True
        
        # Gamma 측정 =======================
        if self.mes_Category == "Gamma":
            if self.gamma_pattern_index >= len(self.gamma_patterns):
                # logging.info("Gamma 측정 완료 → Color Shift로 전환")
                self.mes_Category = "Color Shift"
                self.run_measurement_step()
                return
            
            if self.gray_level_index >= len(self.gray_levels):
                self.gray_level_index = 0
                self.gamma_pattern_index += 1
                self.run_measurement_step()
                return

            pattern = self.gamma_patterns[self.gamma_pattern_index]
            gray_value = self.gray_levels[self.gray_level_index]
            self.current_pattern = pattern

            if pattern == 'white':
                rgb_value = f"{gray_value},{gray_value},{gray_value}"
            elif pattern == 'red':
                rgb_value = f"{gray_value},0,0"
            elif pattern == 'green':
                rgb_value = f"0,{gray_value},0"
            elif pattern == 'blue':
                rgb_value = f"0,0,{gray_value}"            
            
            self.changeColor(rgb_value)

            if self.gray_level_index == 0:
                QTimer.singleShot(3000, self.trigger_measurement)
            else:
                self.trigger_measurement()
                # QTimer.singleShot(2000, self.trigger_measurement)
                
        # Color Shift 측정 =======================
        elif self.mes_Category == "Color Shift":
            if self.colorshift_patterns_index >= len(self.colorshift_patterns):
                # logging.info("Color Shift 측정 완료")
                if hasattr(self, 'measurement_finished_callback') and callable(self.measurement_finished_callback):
                    QTimer.singleShot(200, self.measurement_finished_callback)
                    return
            
            pattern, r, g, b = self.colorshift_patterns[self.colorshift_patterns_index]
            rgb_value = f"{r},{g},{b}"
            self.current_pattern = pattern
            self.changeColor(rgb_value)
            QTimer.singleShot(1000, self.trigger_measurement)
            
    def trigger_measurement(self):
        self.measure_results = {}

        def handle_measure_result(role, result):
            if self.is_stopped:
                logging.info("Result ignored due to stop state")
                self.is_running = False
                return

            if self.is_paused:
                logging.info("Waiting for result due to pause state")
                QTimer.singleShot(500, lambda: handle_measure_result(role, result))
                return

            self.measure_results[role] = result

            if 'main' in self.measure_results and 'sub' in self.measure_results:
                self.process_measure_results()
                if self.mes_Category == "Gamma":
                    self.gray_level_index += 1
                elif self.mes_Category == "Color Shift":
                    self.colorshift_patterns_index += 1

                QTimer.singleShot(200, self.run_measurement_step)

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle_measure_result)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle_measure_result)
            self.sub_measure_thread.start()

    def process_measure_results(self):
        if 'main' in self.measure_results and self.measure_results['main']:
            self.horizontal_angle = "0"
            self.table_widget_main.insertRow(self.row_main)
            self.insert_measure_results_to_table(
                table_widget=self.table_widget_main,
                row=self.row_main,
                win_pattern=self.current_pattern,
                instrument_role='main',
                measured_values=self.measure_results['main']
            )
            self.row_main += 1

            if self.mes_Category == "Gamma":
                self.update_measure_results_to_graph(
                    self.gray_levels[self.gray_level_index],
                    self.measure_results['main'][2],
                    role='main',
                    pattern=self.current_pattern
                )

        if 'sub' in self.measure_results and self.measure_results['sub']:
            self.horizontal_angle = "60"
            self.table_widget_sub.insertRow(self.row_sub)
            self.insert_measure_results_to_table(
                table_widget=self.table_widget_sub,
                row=self.row_sub,
                win_pattern=self.current_pattern,
                instrument_role='sub',
                measured_values=self.measure_results['sub']
            )
            self.row_sub += 1

            if self.mes_Category == "Gamma":
                self.update_measure_results_to_graph(
                    self.gray_levels[self.gray_level_index],
                    self.measure_results['sub'][2],
                    role='sub',
                    pattern=self.current_pattern
                )

    def insert_measure_results_to_table(self, table_widget, row, win_pattern=None, instrument_role='main', measured_values=None):
        if instrument_role == "main":
            instrument_name = self.InstrumentName
        else:
            instrument_name = self.SubInstrumentName

        x, y, lv, cct, duv = measured_values
        u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))
        X, Y, Z = cf.convert_xyz_to_XYZ(float(x), float(y), float(lv))

        col_items = [
            self.mes_Category,
            lv,
            str(float(x)),
            str(float(y)),
            cct.strip(),
            duv,
            str(u_p),
            str(v_p),
            str(X),
            str(Y),
            str(Z),
            win_pattern,
            None,
            f"[{self.ui.vac_lineEdit_red.text()},{self.ui.vac_lineEdit_green.text()},{self.ui.vac_lineEdit_blue.text()}]",
            None, None, None, None, None,
            self.horizontal_angle,
            instrument_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]

        for col, item in enumerate(col_items):
            if item is not None:
                table_widget.setItem(row, col, QTableWidgetItem(item))

        table_widget.scrollToItem(table_widget.item(row, 0))
        QApplication.processEvents()

        return lv, cct, x, y, u_p, v_p, duv

    def update_measure_results_to_graph(self, x_value, y_value, role, pattern):
        ax = self.graph['vac_laboratory']['data_acquisition_system']['output']['ax']
        canvas = self.graph['vac_laboratory']['data_acquisition_system']['output']['canvas']

        line = self.lines_output.get(role, {}).get(pattern)
        if line is None:            
            logging.warning(f"측정 결과를 업데이트할 선이 없습니다: role={role}, pattern={pattern}")
            return

        try:
            x_value = float(x_value)
            y_value = float(y_value)
        except ValueError:
            logging.error(f"그래프에 사용할 수 없는 값입니다: x={x_value}, y={y_value}")
            return

        x_data = list(line.get_xdata())
        y_data = list(line.get_ydata())
        x_data.append(x_value)
        y_data.append(y_value)
        line.set_data(x_data, y_data)

        all_y_values = []
        for role_lines in self.lines_output.values():
            for line_obj in role_lines.values():
                all_y_values.extend(line_obj.get_ydata())

        if all_y_values:
            y_max = max(all_y_values)
            ax.set_ylim(0, y_max * 1.1)

        ax.legend(fontsize=9)
        ax.relim()
        ax.autoscale_view()
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        canvas.draw()
        
    def reset_output_graph(self):
        ax = self.graph['vac_laboratory']['data_acquisition_system']['output']['ax']
        canvas = self.graph['vac_laboratory']['data_acquisition_system']['output']['canvas']

        for role, role_lines in self.lines_output.items():
            for pattern, line_obj in role_lines.items():
                line_obj.set_data([], [])
        
        ax.set_ylim(0, 1)
        legend = ax.get_legend()
        if legend:
            legend.remove()

        ax.relim()
        ax.autoscale_view()
        canvas.draw()
        
    def upload_collected_data_to_DB(self, pk, vac_version):        
        logging.info(f"[Data Collection - Upload] Start uploading measured data: PK = {pk}, VAC Version = {vac_version}")

        self.modelInfo.reset()
        self.modelInfo.Purpose = 'Data Collection'
        self.update_modelInfo()

        self.modelInfo.VAC_Control = 'Y'
        self.modelInfo.VAC_Version = vac_version
        VAC_Info_PK_condition = f"`VAC_Version` = '{vac_version}'"
        self.modelInfo.VAC_Info_PK = cf.search_oneValue_fromDB('W_VAC_Info', 'PK', condition = VAC_Info_PK_condition)

        modelInfo_dic = asdict(self.modelInfo)
        # logging.debug(modelInfo_dic)

        self.measuredData.VAC_SET_Info_PK = cf.insertDictData_To_DB("W_VAC_SET_Info", modelInfo_dic)
        # logging.info(f'VAC_SET_Info_PK = {self.measuredData.VAC_SET_Info_PK}')
        
        self.process_and_upload_measured_data()
        logging.info(f"[Data Collection - Upload] Finished uploading measured data: PK = {pk}, VAC Version = {vac_version}")

    def process_and_upload_measured_data(self):
        data_label = 'data_1'
        
        df_main = self.creat_measurement_df(self.table_widget_main)
        df_sub = self.creat_measurement_df(self.table_widget_sub)

        frontgamma_df = df_main[(df_main['Test'] == 'Gamma') & (df_main['Horizontal_theta'] == 0)]
        sidegamma_df = df_sub[(df_sub['Test'] == 'Gamma') & (df_sub['Horizontal_theta'] == 60)]
        colorshift_df = pd.concat([
                df_main[(df_main['Test'] == 'Color Shift') & (df_main['Horizontal_theta'] == 0)],
                df_sub[(df_sub['Test'] == 'Color Shift') & (df_sub['Horizontal_theta'] == 60)]
            ], ignore_index=True)
        
        frontgamma_df, gamma_avg = self.preprocess_frontgamma(frontgamma_df)
        setattr(self, f'gamma_W_{data_label}', f"{gamma_avg['white']:.2f}")
        setattr(self, f'gamma_R_{data_label}', f"{gamma_avg['red']:.2f}")
        setattr(self, f'gamma_G_{data_label}', f"{gamma_avg['green']:.2f}")
        setattr(self, f'gamma_B_{data_label}', f"{gamma_avg['blue']:.2f}")        
        sidegamma_df, gammalinearity = self.preprocess_sidegamma(sidegamma_df, 'LCM')
        setattr(self, f'gammalinearity_{data_label}', f"{gammalinearity['gammaLinearity']:.2f}")
        colorshift_df, _, _, _, _, _ = self.preprocess_colorshift(colorshift_df)

        transformed_frontgamma_df = self.transform_frontgamma(frontgamma_df, data_label)
        if transformed_frontgamma_df is not None:
            self.load_df_to_DB(transformed_frontgamma_df)

        transformed_gammaLinearity_df = self.transform_gammaLinearity(sidegamma_df, data_label)
        if transformed_gammaLinearity_df is not None:
            self.load_df_to_DB(transformed_gammaLinearity_df)

        transformed_colorshift_df = self.transform_colorshift(colorshift_df, data_label)
        if transformed_colorshift_df is not None:
            self.load_df_to_DB(transformed_colorshift_df)

    def append_data_collection_log(self, pk, vac_version, status, timestamp=None):
        row = self.pk_to_row_index.get(pk)

        vac_version_item = QTableWidgetItem(vac_version)
        status_item = QTableWidgetItem(status)

        if timestamp is None:
            timestamp_item = QTableWidgetItem("")
        else:
            timestamp_item = QTableWidgetItem(timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        if status == "Pending":
            status_item.setBackground(QColor("lightgray"))
        elif status == "Processing":
            status_item.setBackground(QColor("yellow"))
        elif status == "Completed":
            status_item.setBackground(QColor("lightgreen"))
        elif status == "Failed":
            status_item.setBackground(QColor("red"))
        
        self.ui.vac_tableWidget_dataCollectionLog.setItem(row, 0, vac_version_item)
        self.ui.vac_tableWidget_dataCollectionLog.setItem(row, 1, status_item)
        self.ui.vac_tableWidget_dataCollectionLog.setItem(row, 2, timestamp_item)

