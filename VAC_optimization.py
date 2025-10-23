    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        try:
            if metrics and "error" not in metrics:
                logging.info(
                    f"[SPEC(thread)] max|ΔGamma|={metrics['max_dG']:.6f} (≤{metrics['thr_gamma']}), "
                    f"max|ΔCx|={metrics['max_dCx']:.6f}, max|ΔCy|={metrics['max_dCy']:.6f} (≤{metrics['thr_c']})"
                )
            else:
                logging.warning("[SPEC(thread)] evaluation failed — treating as not passed.")

            # 결과 표/차트 갱신
            self._update_spec_views(self._off_store, self._on_store)

            # Step5 애니 정리
            try:
                self.stop_loading_animation(self.label_processing_step_5, self.movie_processing_step_5)
            except Exception:
                pass

            if spec_ok:
                # ✅ 통과: Step5 = complete
                self._step_done(5)
                logging.info("✅ 스펙 통과 — 최적화 종료")
                return

            # ❌ 실패: Step5 = fail
            self._step.fail(5)
            for s in (2,3,4):
                self._step_set_pending(s)

            # 다음 보정 루프
            if iter_idx < max_iters:
                logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
                self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
            else:
                logging.info("⛔ 최대 보정 횟수 도달 — 종료")
        finally:
            self._spec_thread = None
