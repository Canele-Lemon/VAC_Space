import os
import joblib
import logging
import modules.common_func as cf  # 이미 cf.get_normalized_path 사용 중이라 동일하게

def _load_prediction_models(self):
    """
    hybrid_*_model.pkl 파일들을 불러와서 self.models_Y0_bundle에 저장.
    (Gamma / Cx / Cy)
    """
    model_names = {
        "Gamma": "hybrid_Gamma_model.pkl",
        "Cx": "hybrid_Cx_model.pkl",
        "Cy": "hybrid_Cy_model.pkl",
    }

    models_dir = cf.get_normalized_path(__file__, '.', 'models')
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
    return bundle