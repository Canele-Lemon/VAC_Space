def _build_features_for_gray(self, X_dict, gray: int) -> np.ndarray:
    """
    한 행(feature) 구성:
    [ ΔLUT_selected_channels(g), panel_onehot..., frame_rate, model_year,
      gray_norm(=g/255), LUT_index_j(g) ]

    - ΔLUT_selected_channels: self.feature_channels 순서대로, raw 12bit delta @ gray
    - panel_onehot: meta['panel_maker']
    - frame_rate, model_year: meta에서 그대로
    - gray_norm: 0..1
    - LUT_index_j: mapping_j[gray] (0..4095), raw 그대로
    """
    # 1) 소스 참조
    delta_lut = X_dict["lut_delta_raw"]   # dict: ch -> (256,) float32 (raw delta at mapped indices)
    meta      = X_dict["meta"]            # dict: panel_maker(one-hot), frame_rate, model_year
    j_map     = X_dict["mapping_j"]       # (256,) int32, gray -> LUT index(0..4095)

    # 2) 채널 부분: 지정된 feature_channels만 사용 (보통 High 3채널)
    row = []
    for ch in self.feature_channels:
        row.append(float(delta_lut[ch][gray]))   # raw delta (정규화 없음)

    # 3) 메타 부착
    row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
    row.append(float(meta["frame_rate"]))
    row.append(float(meta["model_year"]))

    # 4) gray 위치 정보
    row.append(gray / 255.0)                    # gray_norm

    # 5) LUT 물리 인덱스(매핑) 정보
    j_idx = int(j_map[gray])                    # 0..4095, raw
    row.append(float(j_idx))

    return np.asarray(row, dtype=np.float32)