이제 최적화 flow를 시작해보려고합니다. 아래 start_vac_optimization 메서드 실행으로 시작되는데, 

vac_cmb_ModelYear, vac_cmb_ModelName, vac_cmb_PanelMaker, vac_cmb_FrameRate를 읽어와

"현재 최적화하려는 모델이 몇년도 무슨모델 (패널메이커) 몇hz" 가 맞습니까?"(영어로) 라는 안내창이 뜨고 ok버튼을 누르면 최적화가 시작되도록 하고 싶어요.
​‌
    def start_vac_optimization(self):
        self._spec_policy = VACSpecPolicy()
        
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        self._fine_mode = False
        self._fine_ng_list = None
        
        self._load_jacobian_bundle_npy()
        self._load_prediction_models()
        
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")
        
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self.measure_off_ref_then_on()
    def _load_jacobian_bundle_npy(self):
        """
        bundle["J"]   : (256,3,3)
        bundle["n"]   : (256,)
        bundle["cond"]: (256,)
        """
        if hasattr(self, "_jac_bundle") and self._jac_bundle is not None:
            return
        
        try:
            jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_bundle_ref3008_lam0.001_20251222_142908.npy')
            if not os.path.exists(jac_path):
                raise FileNotFoundError(f"Jacobian npy not found: {jac_path}")

            bundle = np.load(jac_path, allow_pickle=True).item()
            J = np.asarray(bundle["J"], dtype=np.float32) # (256, 3, 3)
            n = np.asarray(bundle["n"], dtype=np.int32)   # (256,)
            cond = np.asarray(bundle["cond"], dtype=np.float32)

            self._jac_bundle = bundle
            self._J_dense = J
            self._J_n = n
            self._J_cond = cond

            logging.info(f"[Jacobian] dense J bundle loaded: {jac_path}, J.shape={J.shape}")

        except Exception:
            logging.exception("[Jacobian] Jacobian load failed")
            raise
        
    def _load_prediction_models(self):
        """
        hybrid_*_model.pkl 파일들을 불러와서 self.models_Y0_bundle에 저장
        (dCx / dCy / dGamma)
        """
        if hasattr(self, "models_Y0_bundle") and self.models_Y0_bundle is not None:
            return
        
        model_names = {
            "dCx": "hybrid_dCx_model.pkl",
            "dCy": "hybrid_dCy_model.pkl",
            "dGamma": "hybrid_dGamma_model.pkl",
        }

        try:
            models_dir = cf.get_normalized_path(__file__, '.', 'models')
            if not os.path.isdir(models_dir):
                raise FileNotFoundError(f"[PredictModel] 모델 디렉터리를 찾을 수 없습니다: {models_dir}")
            
            bundle = {}

            for key, fname in model_names.items():
                path = os.path.join(models_dir, fname)
                
                if not os.path.exists(path):
                    logging.error(f"[PredictModel] 모델 파일을 찾을 수 없습니다: {path}")
                    raise FileNotFoundError(f"Missing model file: {path}")
                
                try:
                    model = joblib.load(path)
                    bundle[key] = model
                    logging.info(f"[PredictModel] {key} 모델 로드 완료: {fname}")
                except Exception as e:
                    logging.exception(f"[PredictModel] {key} 모델 로드 중 오류: {e}")
                    raise

            self.models_Y0_bundle = bundle
            logging.info("[PredictModel] 모든 예측 모델 로드 완료")
            logging.debug(f"[PredictModel] keys: {list(bundle.keys())}")
        
        except Exception:
            raise

추가로 모델 pkl 및 자코비안 파일명 공유드릴게요:
hybrid_dCx_model.pkl
hybrid_dCy_model.pkl
hybrid_dGamma_model.pkl
hybrid_Y1_slope_model.pkl
hybrid_Y2_delta_uv_model.pkl
jacobian_bundle_CSOTCSPI_60Hz_base_lam0.001_20260601_143741.npy
jacobian_bundle_HKCH2_60Hz_base_lam0.001_20260601_143958.npy
jacobian_bundle_HKCH2_120Hz_base_lam0.001_20260601_144220.npy
jacobian_bundle_INX_60Hz_base_lam0.001_20260601_144442.npy
jacobian_bundle_INX_120Hz_base_lam0.001_20260601_144704.npy

