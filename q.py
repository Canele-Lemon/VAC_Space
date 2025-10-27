    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        1. TV setting > VAC OFF → 측정 + UI 업데이트

        2. TV setting > VAC ON → DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 writing → 측정 + UI 업데이트

        3. 측정 결과 평가: 스펙 OK면 종료, NG면 자코비안 행렬을 통해 LUT 보정
        
        4. 보정 LUT TV에 Writing → 측정 + UI 업데이트

        5. 측정 결과 평가: 스펙 OK면 종료, NG면 자코비안 행렬을 통해 LUT 보정
        
        스펙 OK가 나올때까지 LUT 보정 반복...
        """
        base = cf.get_normalized_path(__file__, '..', '..', 'resources/images/pictures')
        self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
        self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
        self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        try:
            # 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            self.debug_print_artifacts(self._jac_artifacts)
            # self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")   # (256, 3K)
            # self.A_Cx    = self._build_A_from_artifacts(artifacts, "Cx")
            # self.A_Cy    = self._build_A_from_artifacts(artifacts, "Cy")
            
            # 예측 모델 로드
            # self.models_Y0_bundle = self._load_prediction_models()

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
        # # 1.2 TV VAC OFF 하기
        # logging.info("[TV CONTROL] TV VAC OFF 전환 시작")
        # if not self._set_vac_active(False):
        #     logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
        #     return
        # logging.info("[TV CONTROL] TV VAC OFF 전환 성공")    
        # # 1.3 OFF 측정 세션 시작
        # logging.info("[MES] VAC OFF 상태 측정 시작")
        # self._run_off_baseline_then_on()
