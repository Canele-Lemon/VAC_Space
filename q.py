def _start_fine_correction_for_ng_list(self, ng_grays, thr_gamma=0.05, thr_c=0.003):
    """
    batch Jacobian 보정 후, NG gray 개수가 10개 이하일 때,
    해당 gray들만 대상으로 per-gray 보정을 수행하는 파인 튜닝 단계.
    """
    # unique + 정렬
    ng_sorted = sorted({int(g) for g in ng_grays})
    if not ng_sorted:
        logging.info("[FINE] NG gray list empty → nothing to do")
        return

    logging.info(f"[FINE] start fine correction session for NG grays: {ng_sorted}")

    # fine 모드 ON
    self._fine_mode = True
    self._fine_ng_list = ng_sorted

    # ON 차트 초기화 (원하면 유지해도 됨)
    self.vac_optimization_gamma_chart.reset_on()
    self.vac_optimization_cie1976_chart.reset_on()

    profile_fine = SessionProfile(
        legend_text="CORR_FINE",
        cie_label=None,
        table_cols={
            "lv":4, "cx":5, "cy":6, "gamma":7,
            "d_cx":8, "d_cy":9, "d_gamma":10
        },
        ref_store=self._off_store
    )

    def _after_fine(store_corr):
        # fine 세션에서 만들어진 ON 데이터를 on_store로 저장
        self._step_done(4)
        self._on_store = store_corr

        # fine 모드 끝 (이후 세션은 per-gray 자동보정 안 함)
        self._fine_mode = False

        # 최종 Spec 평가 한 번 더 (추가 보정은 하지 않기 위해 max_iters=0)
        self._step_start(5)
        self._spec_thread = SpecEvalThread(
            self._off_store,
            self._on_store,
            thr_gamma=thr_gamma,
            thr_c=thr_c,
            parent=self
        )
        self._spec_thread.finished.connect(
            # max_iters=0 → _on_spec_eval_done 안에서 추가 보정 루프 없음
            lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx=0, max_iters=0)
        )
        self._spec_thread.start()

    self._step_start(4)
    self.start_viewing_angle_session(
        profile=profile_fine,
        gray_levels=ng_sorted,          # 🔸 NG gray만 측정
        gamma_patterns=('white',),
        colorshift_patterns=op.colorshift_patterns,
        first_gray_delay_ms=3000,
        cs_settle_ms=1000,
        on_done=_after_fine
    )

def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
    try:
        ng_grays = []
        thr_g = None
        thr_c = None

        if metrics and "error" not in metrics:
            max_dG   = metrics.get("max_dG",  float("nan"))
            max_dCx  = metrics.get("max_dCx", float("nan"))
            max_dCy  = metrics.get("max_dCy", float("nan"))
            thr_g    = metrics.get("thr_gamma", self._spec_thread.thr_gamma if self._spec_thread else None)
            thr_c    = metrics.get("thr_c",     self._spec_thread.thr_c     if self._spec_thread else None)
            ng_grays = metrics.get("ng_grays", [])

            logging.info(
                f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{thr_g}), "
                f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c}), "
                f"NG grays={ng_grays}"
            )
        else:
            logging.warning("[Evaluation] evaluation failed — treating as not passed.")
            ng_grays = []

        # 결과 표/차트 갱신
        self._update_spec_views(iter_idx, self._off_store, self._on_store)

        if spec_ok:
            self._step_done(5)
            logging.info("[Evaluation] Spec 통과 — 최적화 종료")
            return

        # 여기부터 Spec NG
        self._step_fail(5)

        # 🔸 max_iters <= 0 이면 더 이상 자동 보정하지 않고 종료
        #    (fine 단계의 마지막 SpecEval에서 이렇게 들어옴)
        if max_iters <= 0:
            logging.info("[Evaluation] Spec NG but no further auto-correction (max_iters<=0) — 종료")
            return

        # 🔸 batch 단계: NG gray 개수로 branch
        ng_cnt = len(ng_grays)

        if ng_cnt > 0 and ng_cnt <= 10:
            # 10개 이하 → batch 보정 종료, fine 단계로 진입
            logging.info(f"[Evaluation] NG gray {ng_cnt}개 ≤ 10 → batch 보정 종료, per-gray fine correction 시작")
            for s in (2, 3, 4):
                self._step_set_pending(s)

            # threshold가 metrics에 있으면 그대로 사용
            thr_gamma = float(thr_g) if thr_g is not None else 0.05
            thr_c_val = float(thr_c) if thr_c is not None else 0.003

            self._start_fine_correction_for_ng_list(
                ng_grays,
                thr_gamma=thr_gamma,
                thr_c=thr_c_val
            )
            return

        # 🔸 여전히 NG가 많으면 batch jacobian 보정 계속
        if iter_idx < max_iters:
            logging.info(f"[Evaluation] Spec NG — batch 보정 {iter_idx+1}회차 시작")
            for s in (2, 3, 4):
                self._step_set_pending(s)

            thr_gamma = float(thr_g) if thr_g is not None else 0.05
            thr_c_val = float(thr_c) if thr_c is not None else 0.003

            self._run_batch_correction_with_jacobian(
                iter_idx=iter_idx+1,
                max_iters=max_iters,
                thr_gamma=thr_gamma,
                thr_c=thr_c_val,
                metrics=metrics
            )
        else:
            logging.info("[Correction] 최대 보정 횟수 도달 — 종료")

    finally:
        self._spec_thread = None