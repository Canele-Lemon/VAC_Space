def _build_runtime_feature_row_W(
    self,
    lut256_norm: dict,
    gray: int,
    *,
    panel_text: str,
    frame_rate: float,
    model_year_2digit: float = None,
    model_year: float = None,   # ← alias 허용
):
    # --- alias 정리 ---
    if model_year_2digit is None:
        if model_year is not None:
            model_year_2digit = float(int(model_year) % 100)
        else:
            model_year_2digit = 0.0

    row = [
        float(lut256_norm['R_Low'][gray]),  float(lut256_norm['R_High'][gray]),
        float(lut256_norm['G_Low'][gray]),  float(lut256_norm['G_High'][gray]),
        float(lut256_norm['B_Low'][gray]),  float(lut256_norm['B_High'][gray]),
    ]
    row.extend(self._panel_onehot(panel_text).tolist())
    row.append(float(frame_rate))
    row.append(float(model_year_2digit))   # 두 자리 확정
    row.append(gray / 255.0)               # gray_norm
    row.extend([1.0, 0.0, 0.0, 0.0])       # W one-hot
    return np.asarray(row, dtype=np.float32)