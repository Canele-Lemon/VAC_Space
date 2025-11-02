def _do_gray_fix_once(self):
    ctx = self._sess.get('_gray_fix', None)
    if not ctx:
        self._resume_session(); return
    g = int(ctx['g']); tries = ctx['tries']; maxr = ctx['max']
    thr_gamma = float(ctx['thr_gamma']); thr_c = float(ctx['thr_c'])

    if tries >= maxr:
        logging.info(f"[GRAY-FIX] g={g} reached max retries → skip and resume")
        self._sess['_gray_fix'] = None
        self._resume_session()
        return

    ctx['tries'] = tries + 1
    logging.info(f"[GRAY-FIX] g={g} try={ctx['tries']}/{maxr}")

    # ===== 1) Δ 타깃 (해당 g) =====
    # Cx/Cy
    tR = self._off_store['gamma']['main']['white'].get(g, None)
    tO = self._on_store ['gamma']['main']['white'].get(g, None)
    lv_r, cx_r, cy_r = (tR if tR else (np.nan, np.nan, np.nan))
    lv_o, cx_o, cy_o = (tO if tO else (np.nan, np.nan, np.nan))

    dCx = (cx_o - cx_r) if (np.isfinite(cx_o) and np.isfinite(cx_r)) else 0.0
    dCy = (cy_o - cy_r) if (np.isfinite(cy_o) and np.isfinite(cy_r)) else 0.0

    # Gamma(OFF 정규화 프록시)
    #  - ref: OFF 전체로 계산한 gamma (미리 캐시한 self._gamma_off_vec[g])
    #  - on : 현재 gray의 ON 휘도로, OFF 기준 정규화하여 해당 g의 γ 계산
    G_ref_g = float(self._gamma_off_vec[g]) if hasattr(self, "_gamma_off_vec") else np.nan
    G_on_g  = self._gamma_from_off_norm_at_gray(getattr(self, "_off_lv_vec", np.zeros(256)),
                                                lv_on_g=lv_o, g=g)
    dG = (G_on_g - G_ref_g) if (np.isfinite(G_on_g) and np.isfinite(G_ref_g)) else 0.0

    # 데드밴드: 3개 조건 모두 만족하면 보정 없이 재측정만
    if (abs(dCx) <= thr_c) and (abs(dCy) <= thr_c) and (abs(dG) <= thr_gamma):
        logging.info(f"[GRAY-FIX] g={g} within thr (Cx/Cy/Gamma) → remeasure")
        return self._remeasure_same_gray(g)

    # ===== 2) 자코비안 g행 결합 (감마 포함) =====
    # 현장 튜닝: wG_gray는 너무 크지 않게(예: 0.2~0.6) 시작 추천
    wG_gray, wCx, wCy = 0.4, 0.05, 0.5
    Ag = np.vstack([
        wG_gray * self.A_Gamma[g:g+1, :],   # (1,6K)
        wCx     * self.A_Cx   [g:g+1, :],
        wCy     * self.A_Cy   [g:g+1, :],
    ])                                      # (3,6K)
    b  = -np.array([wG_gray*dG, wCx*dCx, wCy*dCy], dtype=np.float32)  # (3,)

    # ===== 3) 리지 해 =====
    ATA = Ag.T @ Ag
    rhs = Ag.T @ b
    lambda_ridge = 1e-3
    ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
    delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

    # ===== 4) 이하(Δh→LUT256→안전 후처리→4096→TV write→read→same gray remeasure) 기존 유지 =====
    ...