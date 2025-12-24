아 잘못말했어요
        except Exception:
            logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
            predicted_vac_data, debug_info = None, None
            predicted_vac_data = base_vac_data

이렇게 해도 되나요?
