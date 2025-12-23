A)
    def _fetch_vac_by_vac_info_pk(self, pk: int):
        """
        `W_VAC_Info` 테이블에서 주어진 `PK` 값으로 `VAC_Version`과 `VAC_Data`를 가져옵니다.
        반환: (vac_version, vac_data) 또는 (None, None)
        """
        try:
            db_conn = pymysql.connect(**config.conn_params)
            cursor = db_conn.cursor()

            cursor.execute("""
                SELECT `VAC_Version`, `VAC_Data`
                FROM `W_VAC_Info`
                WHERE `PK` = %s
            """, (pk,))

            vac_row = cursor.fetchone()

            if not vac_row:
                logging.error(f"[DB] No VAC information found for PK={pk}")
                return None, None

            vac_version = vac_row[0]
            vac_data = vac_row[1]

            logging.info(f"[DB] VAC Info fetched for PK={pk} - Version: {vac_version}")
            return vac_version, vac_data

        except Exception as e:
            logging.error(f"[DB] Error while fetching VAC Info by PK={pk}: {e}")
            return None, None

             이 메서드를 활용해서 Bypass VAC data를 가져온다 했을때 pk=1 입니다.

B) 
처음 최적화 루프 스타트 메서드는 아래와 같고 
    def start_VAC_optimization(self):
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
        self._measure_off_ref_then_on()
         여기서 자코비안 및 예측모델 로드 메서드는 아래와 같습니다:
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
        (Cx / Cy / Gamma)
        """
        if hasattr(self, "models_Y0_bundle") and self.models_Y0_bundle is not None:
            return
        
        model_names = {
            "Cx": "hybrid_dCx_model.pkl",
            "Cy": "hybrid_dCy_model.pkl",
            "Gamma": "hybrid_dGamma_model.pkl",
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

C) 제가 알기에도 3x3 형태가 맞을 텐데 확인을 하려면 어떻게 할까요?
