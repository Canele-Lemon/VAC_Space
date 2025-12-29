def _apply_predicted_vac_and_measure_on(self):
    self._step_start(2)

    BASE_VAC_PK = 3025
    vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(BASE_VAC_PK)
    if base_vac_data is None:
        logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
        return

    base_vac_dict = json.loads(base_vac_data)
    self._vac_dict_cache = base_vac_dict

    try:
        # ✅ 여기서 predicted_vac_json은 "TV write용 문자열"이라고 가정
        predicted_vac_json, new_lut_4096, debug_info = self._generate_predicted_vac_lut(
            base_vac_dict,
            n_iters=2,
            wG=0.4,
            wC=1.0,
            lambda_ridge=1e-3
        )
        if predicted_vac_json is None:
            raise RuntimeError("predicted_vac_json is None")
    except Exception:
        logging.exception("[PredictOpt] 예측 기반 1st 보정 예외 - Base VAC로 진행")
        predicted_vac_json = base_vac_data
        debug_info = None

    # ✅ UI 갱신은 dict로
    predicted_vac_dict = json.loads(predicted_vac_json)
    self._vac_dict_cache = predicted_vac_dict

    lut_dict_plot = {k.replace("channel", "_"): v for k, v in predicted_vac_dict.items() if "channel" in k}
    self._update_lut_chart_and_table(lut_dict_plot)
    self._step_done(2)

    def _after_write(ok, msg):
        if not ok:
            logging.error(f"[VAC Writing] 예측 기반 VAC Writing 실패: {msg} - 최적화 루프 종료")
            return
        logging.info(f"[VAC Writing] 예측 기반 VAC Writing 완료: {msg}")
        logging.info("[VAC Reading] VAC Reading 시작")
        self._read_vac_from_tv(_after_read)

    def _after_read(read_vac_dict):
        self.send_command(self.ser_tv, 'exit')
        if not read_vac_dict:
            logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
            return

        logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
        mismatch_keys = self._verify_vac_data_match(written_data=predicted_vac_dict, read_data=read_vac_dict)
        if mismatch_keys:
            logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
            return

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
            logging.info("[Measurement] 예측 기반 VAC 기준 측정 완료")
            self._step_done(4)

            self._on_store = store_on
            self._update_last_on_lv_norm(store_on)

            logging.info("[Evaluation] Spec 평가 시작")
            self._step_start(5)

            pol = self._spec_policy
            self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
            self._spec_thread.finished.connect(
                lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx=0, max_iters=1)
            )
            self._spec_thread.start()

        logging.info("[Measurement] 예측 기반 VAC 기준 측정 시작")
        self._step_start(4)
        self.start_viewing_angle_session(profile=profile_on, on_done=_after_on)

    logging.info("[VAC Writing] 예측 기반 VAC TV Writing 시작")
    self._write_vac_to_tv(predicted_vac_json, on_finished=_after_write)
    
def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
    """
    조건 1) spec_ok==True: 종료
    조건 2) spec_ok==False and (iter_idx < max_iters): NG Gray batch correction
    """
    try:
        pol = self._spec_policy  # ✅ 여기만

        ng_grays = []
        if metrics and "error" not in metrics:
            max_dG  = metrics.get("max_dG",  float("nan"))
            max_dCx = metrics.get("max_dCx", float("nan"))
            max_dCy = metrics.get("max_dCy", float("nan"))
            ng_grays = metrics.get("ng_grays", [])

            logging.info(
                f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{pol.thr_gamma}), "
                f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{pol.thr_c}), "
                f"NG grays={ng_grays}"
            )
        else:
            logging.warning("[Evaluation] evaluation failed — treating as not passed.")
            ng_grays = []

        self._update_spec_views(iter_idx, self._off_store, self._on_store)

        if spec_ok:
            self._step_done(5)
            logging.info("[Evaluation] Spec 통과 — 최적화 종료")
            self.ui.vac_btn_JSONdownload.setEnabled(True)
            return

        self._step_fail(5)

        if max_iters <= 0:
            logging.info("[Evaluation] Spec NG but no further correction (max_iters<=0) - 종료")
            self.ui.vac_btn_JSONdownload.setEnabled(True)
            return

        if iter_idx >= max_iters:
            logging.info("[Evaluation] Spec NG but 보정 횟수 초과 - 종료")
            self.ui.vac_btn_JSONdownload.setEnabled(True)
            return

        for s in (2, 3, 4):
            self._step_set_pending(s)

        # ✅ thr 넘기지 말고 policy/metrics만 넘겨라
        self._run_batch_correction_with_jacobian(
            iter_idx=iter_idx+1,
            max_iters=max_iters,
            policy=pol,
            metrics=metrics
        )

    finally:
        self._spec_thread = None
        
        =