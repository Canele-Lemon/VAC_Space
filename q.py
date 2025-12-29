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
        
def _run_batch_correction_with_jacobian(self, iter_idx, max_iters, policy: VACSpecPolicy, lam=1e-3, metrics=None):
    logging.info(f"[Batch Correction] iteration {iter_idx} start (Jacobian dense)")

    if not hasattr(self, "_J_dense"):
        logging.error("[Batch Correction] J_dense not loaded")
        return

    self._load_mapping_index_gray_to_lut()

    if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
        logging.error("[Batch Correction] no VAC cache; need latest TV VAC JSON")
        return

    # 1) NG gray 리스트 / Δ 타깃 준비
    if metrics is not None and ("ng_grays" in metrics) and ("dG" in metrics) and ("dCx" in metrics) and ("dCy" in metrics):
        ng_list = list(metrics["ng_grays"])
        d_targets = {
            "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
            "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
            "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
        }
        logging.info(f"[Batch Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
    else:
        dG, dCx, dCy, ng_list, *_ = SpecEvalThread.compute_gray_errors_and_ng_list(
            self._off_store, self._on_store, policy
        )
        d_targets = {
            "Gamma": dG.astype(np.float32),
            "Cx":    dCx.astype(np.float32),
            "Cy":    dCy.astype(np.float32),
        }
        logging.info(f"[Batch Correction] NG grays (recomputed by policy): {ng_list}")

    if not ng_list:
        logging.info("[Batch Correction] no NG gray → 보정 없음")
        return

    # 2) 현재 High LUT 확보
    vac_dict = self._vac_dict_cache
    RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
    GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
    BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

    RH, GH, BH = RH0.copy(), GH0.copy(), BH0.copy()

    # 3) index별 Δ 누적
    delta_acc = {"R": np.zeros_like(RH), "G": np.zeros_like(GH), "B": np.zeros_like(BH)}
    count_acc = {"R": np.zeros_like(RH, dtype=np.int32),
                 "G": np.zeros_like(GH, dtype=np.int32),
                 "B": np.zeros_like(BH, dtype=np.int32)}

    mapLUT = self._mapping_index_gray_to_lut

    n_gray = 256
    dR_gray = np.full(n_gray, np.nan, np.float32)
    dG_gray = np.full(n_gray, np.nan, np.float32)
    dB_gray = np.full(n_gray, np.nan, np.float32)
    corr_flag = np.zeros(n_gray, np.int32)
    wCx_gray = np.full(n_gray, np.nan, np.float32)
    wCy_gray = np.full(n_gray, np.nan, np.float32)
    wG_gray  = np.full(n_gray, np.nan, np.float32)

    step_gain_last = 1.0  # ✅ ng_list가 비어있지 않으니 최소 1회는 들어오지만, 안전하게 초기화

    for g in ng_list:
        if 0 <= g < n_gray:
            corr_flag[g] = 1

        dX = self._solve_delta_rgb_for_gray(
            g,
            d_targets,
            lam=lam,
            thr_c=policy.thr_c,
            thr_gamma=policy.thr_gamma,
            base_wCx=0.5,
            base_wCy=0.5,
            base_wG=1.0,
            boost=3.0,
            keep=0.2,
        )
        if dX is None:
            continue

        dR, dG, dB, wCx_g, wCy_g, wG_g, step_gain = dX
        step_gain_last = step_gain

        dR_gray[g] = dR
        dG_gray[g] = dG
        dB_gray[g] = dB
        wCx_gray[g] = wCx_g
        wCy_gray[g] = wCy_g
        wG_gray[g]  = wG_g

        idx = int(mapLUT[g])
        if 0 <= idx < len(RH):
            delta_acc["R"][idx] += dR
            count_acc["R"][idx] += 1
        if 0 <= idx < len(GH):
            delta_acc["G"][idx] += dG
            count_acc["G"][idx] += 1
        if 0 <= idx < len(BH):
            delta_acc["B"][idx] += dB
            count_acc["B"][idx] += 1

    # 5) 평균 Δ 적용 + clip + monotone
    for ch, arr, arr0 in (("R", RH, RH0), ("G", GH, GH0), ("B", BH, BH0)):
        da = delta_acc[ch]
        ct = count_acc[ch]
        mask = ct > 0
        if not np.any(mask):
            logging.info(f"[Batch Correction] channel {ch}: no indices updated")
            continue
        arr[mask] = arr0[mask] + (da[mask] / ct[mask])
        arr[:] = np.clip(arr, 0.0, 4095.0)
        self._enforce_monotone(arr)

    # 6) 새 LUT 구성
    new_lut_4096 = {
        "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
        "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
        "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
        "RchannelHigh": RH,
        "GchannelHigh": GH,
        "BchannelHigh": BH,
    }
    for k in new_lut_4096:
        arr = np.asarray(new_lut_4096[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        new_lut_4096[k] = np.clip(np.round(arr), 0, 4095).astype(np.uint16)

    # df/log/save는 기존 그대로
    df_corr = self._build_batch_corr_df(
        iter_idx=iter_idx,
        d_targets=d_targets,
        dR_gray=dR_gray, dG_gray=dG_gray, dB_gray=dB_gray,
        corr_flag=corr_flag,
        mapLUT=mapLUT,
        RH0=RH0, GH0=GH0, BH0=BH0,
        RH=RH, GH=GH, BH=BH,
        wCx_gray=wCx_gray, wCy_gray=wCy_gray, wG_gray=wG_gray,
    )
    logging.info(
        f"[Batch Correction] {iter_idx}회차 보정 결과:\n"
        + df_corr.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    self._save_batch_corr_df(iter_idx, df_corr, step_gain=step_gain_last)

    lut_dict_plot = {
        "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
        "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
        "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
    }
    self._update_lut_chart_and_table(lut_dict_plot)

    # 8) TV write → read → re-measure → re-eval
    logging.info(f"[VAC Writing] LUT {iter_idx}차 보정 VAC Data TV Writing start")

    vac_write_json = self.build_vacparam_std_format(
        base_vac_dict=self._vac_dict_cache,
        new_lut_tvkeys=new_lut_4096
    )
    vac_dict_written = json.loads(vac_write_json)
    self._vac_dict_cache = vac_dict_written

    def _after_write(ok, msg):
        logging.info(f"[VAC Writing] write result: {ok} {msg}")
        if not ok:
            return
        logging.info("[VAC Reading] TV reading after write")
        self._read_vac_from_tv(_after_read_back)

    def _after_read_back(vac_dict_after):
        self.send_command(self.ser_tv, 'exit')
        if not vac_dict_after:
            logging.error("[VAC Reading] TV read-back failed")
            return

        mismatch_keys = self._verify_vac_data_match(written_data=vac_dict_written, read_data=vac_dict_after)
        if mismatch_keys:
            logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
            return

        self._step_done(3)

        self._fine_mode = False
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        profile_corr = SessionProfile(
            session_mode=f"CORR #{iter_idx}",
            cie_label=None,
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

        def _after_corr(store_corr):
            self._step_done(4)
            self._on_store = store_corr
            self._update_last_on_lv_norm(store_corr)

            self._step_start(5)
            pol = self._spec_policy
            self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
            self._spec_thread.finished.connect(lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx, max_iters))
            self._spec_thread.start()

        logging.info(f"[Measurement] LUT {iter_idx}차 보정 기준 re-measure start")
        self._step_start(4)
        self.start_viewing_angle_session(
            profile=profile_corr,
            gray_levels=op.gray_levels_256,
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            gamma_settle_ms=1000,
            cs_settle_ms=1000,
            on_done=_after_corr
        )

    self._step_start(3)
    self._write_vac_to_tv(vac_write_json, on_finished=_after_write)