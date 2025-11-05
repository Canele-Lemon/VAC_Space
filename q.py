G_ref_g = float(self._gamma_off_vec[g]) if hasattr(self, "_gamma_off_vec") else np.nan
G_on_g  = self._gamma_from_last_on_norm_at_gray(lv_on_g=lv_o, g=g)
dG = (G_on_g - G_ref_g) if (np.isfinite(G_on_g) and np.isfinite(G_ref_g)) else 0.0