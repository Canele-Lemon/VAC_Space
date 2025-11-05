def _gamma_from_last_on_norm_at_gray(self, lv_on_g: float, g: int) -> float:
    """
    마지막 전체 ON 측정의 Lv0/denom (self._fine_lv0_on / _fine_denom_on)을
    정규화 기준으로 사용하여, 현재 Lv_on(g)에서 γ를 추정.
    """
    lv0  = getattr(self, "_fine_lv0_on", float("nan"))
    denom = getattr(self, "_fine_denom_on", float("nan"))

    if (not np.isfinite(lv0)) or (not np.isfinite(denom)) or denom <= 0:
        return float("nan")

    nor = (lv_on_g - lv0) / denom
    gray_norm = g / 255.0

    if (not np.isfinite(nor)) or nor <= 0:
        return float("nan")
    if gray_norm <= 0 or gray_norm >= 1:
        return float("nan")

    return float(np.log(nor) / np.log(gray_norm))