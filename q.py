def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
    try:
        if metrics and "error" not in metrics:
            max_dG   = metrics.get("max_dG",  float("nan"))
            max_dCx  = metrics.get("max_dCx", float("nan"))
            max_dCy  = metrics.get("max_dCy", float("nan"))
            thr_g    = metrics.get("thr_gamma", self._spec_thread.thr_gamma if self._spec_thread else None)
            thr_c    = metrics.get("thr_c",     self._spec_thread.thr_c     if self._spec_thread else None)
            ng_grays = metrics.get("ng_grays", [])

            logging.info(
                f"[SPEC(thread)] max|ΔGamma|={max_dG:.6f} (≤{thr_g}), "
                f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c}), "
                f"NG grays={ng_grays}"
            )
        else:
            logging.warning("[SPEC(thread)] evaluation failed — treating as not passed.")
            ng_grays = []

        # 📊 결과 표/차트 갱신 (기존 유지)
        self._update_spec_views(iter_idx, self._off_store, self._on_store)

        # ✅ 스펙 통과: NG gray 없음
        if spec_ok:
            self._step_done(5)
            logging.info("✅ 스펙 통과 — 최적화 종료")
            return

        # ❌ 스펙 실패
        self._step_fail(5)

        # 🔻 여기서부터는 NG gray 리스트를 활용하는 지점입니다.
        #     1차 구현: 기존처럼 full-frame correction을 돌리되,
        #     나중에 자코비안 기반 일괄 보정 함수를 여기에서 호출하면 됩니다.

        # (예시 1) 지금 구조 유지: 예전처럼 전체 보정 루프
        if iter_idx < max_iters:
            logging.info(
                f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1}), "
                f"NG grays={ng_grays}"
            )
            for s in (2, 3, 4):
                self._step_set_pending(s)
            self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
        else:
            logging.info("⛔ 최대 보정 횟수 도달 — 종료")

        # (예시 2) 나중에 자코비안 기반 '한 번에 보정'을 도입하면,
        #          여기서 아래처럼 별도 함수를 호출하면 됩니다.
        # if not spec_ok:
        #     self._run_jacobian_batch_correction(ng_grays, metrics)

    finally:
        self._spec_thread = None