def _apply_predicted_vac_and_measure_on(self):
    """
    1) DB에서 base VAC JSON 로딩
    2) 예측 모델 + Jacobian으로 1st 보정 VAC 생성
    3) 보정된 VAC를 TV에 write
    4) write/read 검증 후 VAC ON 측정 및 spec 평가
    """
    self._step_start(2)

    # 1) DB에서 base VAC JSON 로딩 (예: PK=2)
    vac_version, vac_data_base = self._fetch_vac_by_vac_info_pk(2)
    if vac_data_base is None:
        logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
        return

    # 2) 예측 모델 + Jacobian 기반 1차 보정
    #    _predictive_first_optimize는 예측+보정까지 해서
    #    "보정된 VAC JSON 문자열"을 돌려주는 함수라고 가정
    try:
        predicted_vac_json, debug_info = self._predictive_first_optimize(
            vac_data_base,
            n_iters=2,
            wG=0.4,
            wC=1.0,
            lambda_ridge=1e-3
        )
    except Exception:
        logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
        predicted_vac_json, debug_info = None, None

    if predicted_vac_json is None:
        logging.warning("[PredictOpt] 예측 기반 보정 실패 또는 비활성 - Base VAC로 진행합니다.")
        vac_json_to_use = vac_data_base
    else:
        vac_json_to_use = predicted_vac_json

    # dict로 파싱해서 UI/검증에 사용
    vac_dict_to_use = json.loads(vac_json_to_use)
    self._vac_dict_cache = vac_dict_to_use

    # LUT 차트/테이블 업데이트용 (4096 LUT → plot용 key rename)
    lut_dict_plot = {
        key.replace("channel", "_"): v
        for key, v in vac_dict_to_use.items()
        if "channel" in key
    }
    self._update_lut_chart_and_table(lut_dict_plot)
    self._step_done(2)

    # 이후 비동기 콜백에서 참조할 수 있도록 클로저 변수로 잡아둠
    written_vac_dict = vac_dict_to_use
    vac_json_for_tv = vac_json_to_use

    # 3) TV에 VAC write
    logging.info("[VAC Writing] Predicted/Base VAC TV Writing 시작")
    self._write_vac_to_tv(vac_json_for_tv, on_finished=_after_write)

    # ─────────────────────────────────────────
    #   내부 콜백 정의 (_after_write, _after_read, _after_on)
    # ─────────────────────────────────────────
    def _after_write(ok, msg):
        if not ok:
            logging.error(f"[VAC Writing] Predicted/Base VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
            return
        
        logging.info(f"[VAC Writing] Predicted/Base VAC 데이터 Writing 완료: {msg}")
        logging.info("[VAC Reading] VAC Reading 시작")
        self._read_vac_from_tv(_after_read)

    def _after_read(read_vac_dict):
        # panelcontroller 재시작 (기존 로직 유지)
        self.send_command(self.ser_tv, 'restart panelcontroller')
        time.sleep(1.0)
        self.send_command(self.ser_tv, 'restart panelcontroller')
        time.sleep(1.0)
        self.send_command(self.ser_tv, 'exit')

        if not read_vac_dict:
            logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
            return

        logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
        mismatch_keys = self._verify_vac_data_match(
            written_data=written_vac_dict,   # ← 예측 보정된 dict 기준으로 비교
            read_data=read_vac_dict
        )

        if mismatch_keys:
            logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
            return
        else:
            logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")

        self._step_done(3)

        # 미세보정 플래그 초기화
        self._fine_mode = False
        
        # ON 차트 초기화
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        # VAC ON 측정 프로파일 (OFF 결과를 ref로 사용)
        profile_on = SessionProfile(
            legend_text="VAC ON (Predicted)",
            cie_label="data_2",
            table_cols={
                "lv":4, "cx":5, "cy":6, "gamma":7,
                "d_cx":8, "d_cy":9, "d_gamma":10
            },
            ref_store=self._off_store
        )

        def _after_on(store_on):
            logging.info("[Measurement] Predicted/Base VAC 기준 VAC ON 측정 완료")
            self._step_done(4)
            self._on_store = store_on
            self._update_last_on_lv_norm(store_on)
            
            logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
            self._step_start(5)
            self._spec_thread = SpecEvalThread(
                self._off_store,
                self._on_store,
                thr_gamma=0.05,
                thr_c=0.003,
                parent=self
            )
            # iter_idx / max_iters는 1차 보정 loop에서는 (0, 0)으로 둬도 OK
            self._spec_thread.finished.connect(
                lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=0)
            )
            self._spec_thread.start()

        logging.info("[Measurement] Predicted/Base VAC 기준 VAC ON 측정 시작")
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