def _is_gray_spec_ok(self, gray:int, *, thr_gamma=0.05, thr_c=0.003,
                     off_store=None, on_store=None) -> bool:
    off_store = off_store if off_store is not None else self._off_store
    on_store  = on_store  if on_store  is not None else self._on_store
    ref = off_store['gamma']['main']['white'].get(gray, None)
    on  = on_store ['gamma']['main']['white'].get(gray, None)
    if not ref or not on:
        return True
    lv_r, cx_r, cy_r = ref
    lv_o, cx_o, cy_o = on

    dCx = abs(cx_o - cx_r) if (np.isfinite(cx_o) and np.isfinite(cx_r)) else 0.0
    dCy = abs(cy_o - cy_r) if (np.isfinite(cy_o) and np.isfinite(cy_r)) else 0.0

    # Gamma(OFF 정규화 프록시)
    if hasattr(self, "_gamma_off_vec") and hasattr(self, "_off_lv_vec"):
        G_ref_g = float(self._gamma_off_vec[gray])
        G_on_g  = self._gamma_from_off_norm_at_gray(self._off_lv_vec, lv_on_g=lv_o, g=gray)
        dG = abs(G_on_g - G_ref_g) if (np.isfinite(G_on_g) and np.isfinite(G_ref_g)) else 0.0
    else:
        dG = 0.0

    return (dCx <= thr_c) and (dCy <= thr_c) and (dG <= thr_gamma)