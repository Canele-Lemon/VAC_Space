def _is_gray_spec_ok(self, gray:int, *, thr_gamma=0.05, thr_c=0.003) -> bool:
    # OFF 레퍼런스
    ref = self._off_store['gamma']['main']['white'].get(gray, None)
    on  = self._on_store ['gamma']['main']['white'].get(gray, None)
    if not ref or not on:
        return True  # 데이터 없으면 패스 취급(측정 실패는 상위 로직에서 처리)
    lv_r, cx_r, cy_r = ref
    lv_o, cx_o, cy_o = on

    # 감마 한 점 계산(안전 가드: 전체 벡터 재계산보다 간단 추정)
    # 정확도를 높이려면 기존 _compute_gamma_series로 전체 재계산 후 gray 인덱스 꺼내도 됩니다.
    # 여기서는 간단화를 위해 _compute_gamma_series 사용:
    def _one_gamma(store):
        lv = np.zeros(256); 
        for g in range(256):
            t = store['gamma']['main']['white'].get(g, None)
            lv[g] = float(t[0]) if t else np.nan
        return self._compute_gamma_series(lv)

    G_ref = _one_gamma(self._off_store)
    G_on  = _one_gamma(self._on_store)
    dG  = abs(G_on[gray]) if np.isfinite(G_on[gray]) and np.isfinite(G_ref[gray]) else 0.0
    dCx = abs(cx_o - cx_r) if np.isfinite(cx_o) and np.isfinite(cx_r) else 0.0
    dCy = abs(cy_o - cy_r) if np.isfinite(cy_o) and np.isfinite(cy_r) else 0.0

    return (dG <= thr_gamma) and (dCx <= thr_c) and (dCy <= thr_c)