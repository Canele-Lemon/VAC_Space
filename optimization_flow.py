    def _get_jacobian_filename(self):
        panel_maker = self.ui.vac_cmb_PanelMaker.currentText().strip()
        frame_rate = self.ui.vac_cmb_FrameRate.currentText().strip().replace("Hz", "").strip()

        panel_maker = panel_maker.replace("(", "").replace(")", "")

        models_dir = cf.get_normalized_path(__file__, '.', 'models')

        pattern = f"jacobian_bundle_{panel_maker}_{frame_rate}Hz_*.npy"

        search_path = os.path.join(models_dir, pattern)

        matched_files = glob.glob(search_path)

        if not matched_files:
            raise FileNotFoundError(
                f"No Jacobian file matched for PanelMaker={panel_maker}, FrameRate={frame_rate}Hz"
            )

        matched_files.sort()
        return os.path.basename(matched_files[-1])

    def _load_jacobian_bundle_npy(self):
        """
        bundle["J"]   : (256,3,3)
        bundle["n"]   : (256,)
        bundle["cond"]: (256,)
        """
        if hasattr(self, "_jac_bundle") and self._jac_bundle is not None:
            return
        
        try:
            jac_filename = self._get_jacobian_filename()
            jac_path = cf.get_normalized_path(__file__, '.', 'models', jac_filename)
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

현재 이렇게 코드를 수정했는데, 사용자가 선택한 모델에 따른 아티팩트 로드할때 아래와 같은 규칙이 필요해요.
panel maker == CSOT 외 'CSOT'가 포함된 모든 문자인 경우 CSOT(CSPI)와 같은 것입니다.
panel maker == HKC(H5) 외 'HKC'가 포함된 모든 문자인 경우 HKC(H2)와 같은 것입니다.
panel maker == 'BOE'가 포함된 모든 문자인 경우 120HZ면 INX 60HZ와 같은 거고 60HZ인 경우 HKC와 같은 것입니다.

