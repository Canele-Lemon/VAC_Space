vac write가 잘 안되는 경우가 있어서, _write_vac_to_tv를 강화하려고 해요

    def _apply_vac_from_db_and_measure_on(self):
        self._step_start(2)
        
        # panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        # fr = self.ui.vac_cmb_FrameRate.currentText().strip()
        # vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        # if vac_data is None:
        #     logging.error(f"[DB] {panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
        #     return

        vac_version, vac_data = self._fetch_vac_by_vac_info_pk(2582)
        if vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        vac_dict = json.loads(vac_data)
        self._vac_dict_cache = vac_dict
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in vac_dict.items() if "channel" in key}
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] DB fetch VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] DB fetch VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=vac_dict, read_data=read_vac_dict)

            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")

            self._step_done(3)

            self._fine_mode = False
            
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_on = SessionProfile(
                legend_text="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=5))
                self._spec_thread.start()

            logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                gamma_settle_ms=1000,
                cs_settle_ms=1000,
                on_done=_after_on
            )

        logging.info("[VAC Writing] DB fetch VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(vac_data, on_finished=_after_write)
        
    def _write_vac_to_tv(self, vac_data, on_finished):
        self._step_start(3)
        t = WriteVACdataThread(parent=self, ser_tv=self.ser_tv,
                                vacdataName=self.vacdataName, vacdata_loaded=vac_data)
        t.write_finished.connect(lambda ok, msg: on_finished(ok, msg))
        t.start()

    def _read_vac_from_tv(self, on_finished):
        t = ReadVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName)
        t.data_read.connect(lambda data: on_finished(data))
        t.error_occurred.connect(lambda err: (logging.error(err), on_finished(None)))
        t.start()
보통 write 하고 read 하는데 새로고침이 안됐다고 해야 하나 vac가 안입혀져 있는 오류가 10번중 2번은 발생하는거같아요. 이를 위해 
class WriteVACdataThread(QThread):
    write_finished = Signal(bool, str)

    def __init__(self, parent, ser_tv, vacdataName, vacdata_loaded):
        super().__init__(parent)
        self.parent = parent
        self.ser_tv = ser_tv
        self.vacdataName = vacdataName
        self.vacdata_loaded = vacdata_loaded

    def run(self):
        try:
            vac_debug_path = "/mnt/lg/cmn_data/panelcontroller/db/vac_debug"
            self.parent.send_command(self.ser_tv, 's')
            output = self.parent.check_directory_exists(vac_debug_path)

            if output == 'exists':
                pass
            elif output == 'not_exists':
                self.parent.send_command(self.ser_tv, f"mkdir -p {vac_debug_path}")
            else:
                self.parent.send_command(self.ser_tv, 'exit')
                self.write_finished.emit(False, f"Error checking VAC debug path: {output}")
                return

            copyVACdata = f"cp /etc/panelcontroller/db/vac/{self.vacdataName} {vac_debug_path}"
            self.parent.send_command(self.ser_tv, copyVACdata)

            if self.vacdata_loaded is None:
                self.parent.send_command(self.ser_tv, 'exit')
                self.write_finished.emit(False, "No VAC data loaded.")
                return
            
            writeVACdata = f'cat > {vac_debug_path}/{self.vacdataName}'
            self.ser_tv.write((writeVACdata + '\n').encode())
            time.sleep(0.1)
            self.ser_tv.write(self.vacdata_loaded.encode())
            time.sleep(0.1)
            self.ser_tv.write(b'\x04')  # Ctrl+D
            time.sleep(0.1)
            self.ser_tv.write(b'\x04')  # Ctrl+D
            time.sleep(0.1)
            self.ser_tv.flush()

            self.parent.read_output(self.ser_tv, output_limit=1000)

            self.parent.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.parent.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.parent.send_command(self.ser_tv, 'exit')

            self.write_finished.emit(True, f"VAC data written to {vac_debug_path}/{self.vacdataName}")
        except Exception as e:
            self.write_finished.emit(False, f"Unexpected error while writing VAC data: {e}")
여기서 새로고침과 같은 self.parent.send_command(self.ser_tv, 'restart panelcontroller')를 두 번 했는데도 방금 또 그랬어요.
그래서 read한 다음에도 한번 더 send_command(self.ser_tv, 'restart panelcontroller')를 넣어주려고 합니다. 어디에 넣으면 좋을까요?
