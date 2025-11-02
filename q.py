def _gamma_from_off_norm_at_gray(self, off_lv_vec_256, lv_on_g: float, g: int) -> float:
    """
    OFF 시리즈로 정규화 기준을 삼아 해당 gray의 ON 휘도로 γ를 계산.
    gamma(g) = log(Ynorm) / log(g/255)
      where Ynorm = (Lv_on(g) - Lv_off(0)) / max(Lv_off[1:]-Lv_off(0))
    g==0,255 또는 Ynorm<=0 이면 np.nan
    """
    if g <= 0 or g >= 255:
        return np.nan
    lv0 = float(off_lv_vec_256[0])
    denom = float(np.nanmax(off_lv_vec_256[1:] - lv0))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    yn = (float(lv_on_g) - lv0) / denom
    if not np.isfinite(yn) or yn <= 0:
        return np.nan
    gn = g / 255.0
    return float(np.log(yn) / np.log(gn)) if gn > 0 else np.nan