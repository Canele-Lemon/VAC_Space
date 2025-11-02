def _is_gray_spec_ok(self, gray:int, *, thr_gamma=0.05, thr_c=0.003,
                     off_store=None, on_store=None) -> bool:
    off_store = off_store or self._off_store
    on_store  = on_store  or self._on_store  # 종료 후 호출도 호환

    ref = off_store['gamma']['main']['white'].get(gray, None)
    on  = on_store ['gamma']['main']['white'].get(gray, None)
    if not ref or not on:
        return True

    # 감마는 전체 벡터로 계산 후 인덱싱
    def _gamma_from(store):
        lv = np.array([store['gamma']['main']['white'].get(i,(np.nan,)*3)[0] for i in range(256)],
                      dtype=float)
        return self._compute_gamma_series(lv)

    G_ref = _gamma_from(off_store)
    G_on  = _gamma_from(on_store)

    lv_r, cx_r, cy_r = ref
    lv_o, cx_o, cy_o = on
    dG  = abs(G_on[gray] - G_ref[gray]) if np.isfinite(G_on[gray]) and np.isfinite(G_ref[gray]) else 0.0
    dCx = abs(cx_o - cx_r) if np.isfinite(cx_o) and np.isfinite(cx_r) else 0.0
    dCy = abs(cy_o - cy_r) if np.isfinite(cy_o) and np.isfinite(cy_r) else 0.0
    return (dG <= thr_gamma) and (dCx <= thr_c) and (dCy <= thr_c)