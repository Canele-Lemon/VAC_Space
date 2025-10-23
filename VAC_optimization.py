def start_VAC_optimization(self):
    """
    ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
    전체 Flow:
    [STEP 1] TV setting > VAC OFF → 측정(OFF baseline)
    [STEP 2] DB LUT fetch → 예측/보정
    [STEP 3] VAC ON 전환 → 보정 LUT 적용
    [STEP 4] 보정 LUT 기준 측정
    [STEP 5] 결과 평가
    """
    base = cf.get_normalized_path(__file__, '..', '..', 'resources/images/pictures')
    self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
    self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
    self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))

    # ✅ 시작 시 모든 단계 라벨을 '대기' 아이콘으로 세팅(라벨 크기에 맞춰 스케일)
    self._set_icon_scaled(self.ui.vac_label_pixmap_step_1, self.process_pending_pixmap)
    self._set_icon_scaled(self.ui.vac_label_pixmap_step_2, self.process_pending_pixmap)
    self._set_icon_scaled(self.ui.vac_label_pixmap_step_3, self.process_pending_pixmap)
    self._set_icon_scaled(self.ui.vac_label_pixmap_step_4, self.process_pending_pixmap)
    self._set_icon_scaled(self.ui.vac_label_pixmap_step_5, self.process_pending_pixmap)

    # (기존) 애니 시작 선언이 있다면 유지
    self._step_start(1)

    try:
        # 자코비안 로드
        artifacts = self._load_jacobian_artifacts()
        self._jac_artifacts = artifacts
        self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")
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
    self._off_store = {
        'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}},
                  'sub':  {'white':{},'red':{},'green':{},'blue':{}}},
        'colorshift': {'main': [], 'sub': []}
    }
    self._on_store = {
        'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}},
                  'sub':  {'white':{},'red':{},'green':{},'blue':{}}},
        'colorshift': {'main': [], 'sub': []}
    }

    logging.info("[TV CONTROL] TV VAC OFF 전환 시작")
    if not self._set_vac_active(False):
        logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
        return
    logging.info("[TV CONTROL] TV VAC OFF 전환 성공")
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

        # Step5 애니 정리
        try:
            self.stop_loading_animation(self.label_processing_step_5, self.movie_processing_step_5)
        except Exception:
            pass

        if spec_ok:
            # ✅ 통과: Step5 = complete
            self._set_icon_scaled(self.ui.vac_label_pixmap_step_5, self.process_complete_pixmap)
            logging.info("✅ 스펙 통과 — 최적화 종료")
            return

        # ❌ 실패: Step5 = fail
        self._set_icon_scaled(self.ui.vac_label_pixmap_step_5, self.process_fail_pixmap)

        # 요구사항: 실패시에만 Step2~4 아이콘을 pending으로 되돌림
        self._set_icon_scaled(self.ui.vac_label_pixmap_step_2, self.process_pending_pixmap)
        self._set_icon_scaled(self.ui.vac_label_pixmap_step_3, self.process_pending_pixmap)
        self._set_icon_scaled(self.ui.vac_label_pixmap_step_4, self.process_pending_pixmap)

        # 다음 보정 루프
        if iter_idx < max_iters:
            logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
            self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
        else:
            logging.info("⛔ 최대 보정 횟수 도달 — 종료")
    finally:
        self._spec_thread = None