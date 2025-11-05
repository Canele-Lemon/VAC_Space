# Gamma(OFF 기준 타깃 vs 마지막 ON norm 기준 ON γ)
if hasattr(self, "_gamma_off_vec") and hasattr(self, "_fine_lv0_on") and hasattr(self, "_fine_denom_on"):
    G_ref_g = float(self._gamma_off_vec[gray])  # 여전히 VAC OFF 기준 γ 타깃
    G_on_g  = self._gamma_from_last_on_norm_at_gray(lv_on_g=lv_o, g=gray)
    dG = abs(G_on_g - G_ref_g) if (np.isfinite(G_on_g) and np.isfinite(G_ref_g)) else 0.0
else:
    dG = 0.0