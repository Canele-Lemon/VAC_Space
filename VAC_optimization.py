    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        [STEP 1] TV setting > VAC OFF → 측정(OFF baseline) + UI 업데이트

        [STEP 2] TV setting > VAC OFF → DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 writing → 측정(ON 현재) + UI 업데이트

        [STEP 3] 스펙 확인 → 통과면 종료
        
        [STEP 4] 미통과면 자코비안 기반 보정(256기준) → 4096 보간 반영 → 예측모델 검증 → OK면 → TV 적용 → 재측정 → 스펙 재확인
        [STEP 5] (필요 시 반복 2~3회만)
        """
        base = cf.get_normalized_path(__file__, '..', '..', 'resources/images/pictures')
        self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
        self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
        self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))
        self._step_start(1)
        try:
            # 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")   # (256, 3K)
            self.A_Cx    = self._build_A_from_artifacts(artifacts, "Cx")
            self.A_Cy    = self._build_A_from_artifacts(artifacts, "Cy")
            
            # 예측 모델 로드
            self.models_Y0_bundle = self._load_prediction_models()

        except FileNotFoundError as e:
            logging.error(f"[VAC Optimization] Jacobian file not found: {e}")

        except KeyError as e:
            logging.error(f"[VAC Optimization] Missing key in artifacts: {e}")

        except Exception as e:
            logging.exception("[VAC Optimization] Unexpected error occurred")

        # 1. VAC OFF 보장 + 측정
        # 1.1 결과 저장용 버퍼 초기화 (OFF / ON 구분)
        self._off_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        self._on_store  = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        # 1.2 TV VAC OFF 하기
        logging.info("[TV CONTROL] TV VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV CONTROL] TV VAC OFF 전환 성공")    
        # 1.3 OFF 측정 세션 시작
        logging.info("[MES] VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()
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
