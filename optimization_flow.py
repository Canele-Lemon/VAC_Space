# 5) meta (panel onehot + frame_rate)
# 학습 feature_schema:
# ΔLUT(6ch) + panel_maker_onehot + frame_rate + gray_norm + LUT_j
# model_year는 학습에서 제외했으므로 예측 X에도 넣지 않는다.

artifact_panel, artifact_hz = self._resolve_artifact_key()

panel_text = artifact_panel
frame_rate = float(artifact_hz)

panel_onehot = self.panel_text_to_onehot(panel_text).astype(np.float32)

logging.debug(
    f"[Predict META] panel_text={panel_text}, frame_rate={frame_rate}, "
    f"panel_onehot_dim={len(panel_onehot)}"
)

# frame_rate
row.append(float(frame_rate))

# gray_norm, LUT_j
row.append(float(g / 255.0))
row.append(float(idx_map[g]))


def _predict_y0(d_lut_256: dict, pat: str = "W"):
    X = _build_X_y0_per_gray(d_lut_256, pat=pat)

    try:
        expected = self.models_Y0_bundle["dCx"]["linear_model"].named_steps["scaler"].n_features_in_
        logging.debug(f"[Predict X] X.shape={X.shape}, expected_features={expected}")
    except Exception:
        logging.exception("[Predict X] failed to check expected feature count")

    ...
    
    
    
