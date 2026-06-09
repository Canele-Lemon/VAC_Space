2026-06-09 19:13:28,531 - INFO - subpage_vacspace.py:2718 - [TV Control] VAC 적용 상태: True
2026-06-09 19:13:28,531 - INFO - subpage_vacspace.py:786 - [TV Control] VAC ON 전환 성공
2026-06-09 19:13:28,531 - INFO - subpage_vacspace.py:788 - [Measurement] VAC ON 측정 시작
2026-06-09 19:13:28,578 - INFO - subpage_vacspace.py:2591 - [DB] VAC Info fetched for PK=3025 - Version: LUT_3_Base
2026-06-09 19:13:28,592 - INFO - subpage_vacspace.py:2671 - [Correction] loaded d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\data\LUT_index_mapping.csv, shape=(256, 2), range=0~4095
2026-06-09 19:13:28,690 - INFO - subpage_vacspace.py:2591 - [DB] VAC Info fetched for PK=1 - Version: bypass
2026-06-09 19:13:28,696 - DEBUG - subpage_vacspace.py:3273 - [UI META] panel_maker='INX', frame_rate='120.0Hz', model_year='Y25'
2026-06-09 19:13:28,699 - ERROR - subpage_vacspace.py:1586 - [PredictOpt] failed
Traceback (most recent call last):
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1489, in _generate_predicted_vac_lut
    dCx_pred, dCy_pred, dGamma_pred = _predict_y0(d_lut_256, pat=pat)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1472, in _predict_y0
    dCx_pred    = _hybrid_predict(self.models_Y0_bundle["dCx"], X)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1462, in _hybrid_predict
    base_s = lm.predict(X).astype(np.float32)
             ^^^^^^^^^^^^^
  File "D:\DEV\gayasan\he_opticalmeasurement\.venv\Lib\site-packages\sklearn\pipeline.py", line 796, in predict
    Xt = transform.transform(Xt)
         ^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\DEV\gayasan\he_opticalmeasurement\.venv\Lib\site-packages\sklearn\utils\_set_output.py", line 319, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\DEV\gayasan\he_opticalmeasurement\.venv\Lib\site-packages\sklearn\preprocessing\_data.py", line 1111, in transform
    X = validate_data(
        ^^^^^^^^^^^^^^
  File "D:\DEV\gayasan\he_opticalmeasurement\.venv\Lib\site-packages\sklearn\utils\validation.py", line 3059, in validate_data
    _check_n_features(_estimator, X, reset=reset)
  File "D:\DEV\gayasan\he_opticalmeasurement\.venv\Lib\site-packages\sklearn\utils\validation.py", line 2923, in _check_n_features
    raise ValueError(
ValueError: X has 15 features, but StandardScaler is expecting 14 features as input.
2026-06-09 19:13:28,700 - ERROR - subpage_vacspace.py:819 - [PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행
Traceback (most recent call last):
  File "d:\DEV\gayasan\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 817, in apply_predicted_vac_and_measure_on
    raise RuntimeError("predicted_vac_data is None")
RuntimeError: predicted_vac_data is None

위와 같은 에러가 발생했는데 어떻게 해결해야 할지 봐주시겠어요?

현재 코드:

    def apply_predicted_vac_and_measure_on(self):
        self._step_start(2)
        
        BASE_VAC_PK = 3025
        vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(BASE_VAC_PK)
        if base_vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        base_vac_dict = json.loads(base_vac_data)
        self._vac_dict_cache = base_vac_dict
        
        try:
            predicted_vac_data, new_lut_4096, debug_info = self._generate_predicted_vac_lut(
                base_vac_dict,
                n_iters=1,
                wG=0.4,
                wC=1.0,
                lambda_ridge=1e-3
            )
            if predicted_vac_data is None:
                raise RuntimeError("predicted_vac_data is None")
        except Exception:
            logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
            predicted_vac_data = base_vac_data
            debug_info = None
            
        predicted_vac_dict = json.loads(predicted_vac_data)
        self._vac_dict_cache = predicted_vac_dict
            
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in predicted_vac_dict.items() if "channel" in key}
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            self.send_command(self.ser_tv, 'exit')
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self.verify_vac_data_match(written_data=predicted_vac_dict, read_data=read_vac_dict)

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
                session_mode="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                pol = self._spec_policy
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self.on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=1))
                self._spec_thread.start()

            logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                on_done=_after_on
            )

        # logging.info("[VAC Writing] 예측기반 최적화 VAC 데이터 TV Writing 시작")
        # self._write_vac_to_tv(predicted_vac_data, on_finished=_after_write)
