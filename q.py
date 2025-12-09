    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        """
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        self._fine_mode = False
        self._fine_ng_list = None
        
        try:
            self._load_jacobian_bundle_npy()
        except Exception as e:
            logging.exception("[Jacobian] Jacobian load failed")
            return
        
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")
        
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()
이 최적화 루프의 시나리오가
최적화 시작->VAC OFF 측정->Base VAC기준 결과 예측<->보정 수행 -> tv VAC 적용 -> 측정 -> spen in 평가 -> ng 시 미세보정 -> 측정 -> 평가 -> ng 시 미세보정 -> ... 이런식으로 돌아갑니다. 현재 Base VAC기준 결과 예측<->보정 수행 부분은 base VAC만 불러오는 것만 수행하는데 학습한 예측 모델을 불러와 예측<->보정을 하도록 메서드를 만들어주세요. 모델 경로는 아래와 같아요:
Cx_model_path = cf.get_normalized_path(__file__, '.', 'models', 'hybrid_Cx_model.pkl')
Cy_model_path = cf.get_normalized_path(__file__, '.', 'models', 'hybrid_Cy_model.pkl')
Gamma_model_path = cf.get_normalized_path(__file__, '.', 'models', 'hybrid_Gamma_model.pkl')
    def _apply_vac_from_db_and_measure_on(self):
        self._step_start(2)
        
        # panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        # fr = self.ui.vac_cmb_FrameRate.currentText().strip()
        # vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        # if vac_data is None:
        #     logging.error(f"[DB] {panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
        #     return

        vac_version, vac_data = self._fetch_vac_by_vac_info_pk(2)
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
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'exit')
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
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=0))
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
