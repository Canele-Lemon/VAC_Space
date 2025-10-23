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

        # Step5: 애니 정리 후 아이콘 설정
        try:
            self.stop_loading_animation(self.label_processing_step_5, self.movie_processing_step_5)
        except Exception:
            pass

        if spec_ok:
            # ✅ 통과 → Step5 = complete, 나머지는 손대지 않음
            self.ui.vac_label_pixmap_step_5.setPixmap(self.process_complete_pixmap)
            logging.info("✅ 스펙 통과 — 최적화 종료")
            return

        # ❌ 실패 → Step5 = fail
        self.ui.vac_label_pixmap_step_5.setPixmap(self.process_fail_pixmap)

        # ⇦ 요구사항: 실패시에만 Step2~4를 다시 '대기'로 되돌림 (아이콘만 교체)
        self.ui.vac_label_pixmap_step_2.setPixmap(self.process_pending_pixmap)
        self.ui.vac_label_pixmap_step_3.setPixmap(self.process_pending_pixmap)
        self.ui.vac_label_pixmap_step_4.setPixmap(self.process_pending_pixmap)

        # 다음 보정 루프 진행
        if iter_idx < max_iters:
            logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
            self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
        else:
            logging.info("⛔ 최대 보정 횟수 도달 — 종료")

    finally:
        self._spec_thread = None