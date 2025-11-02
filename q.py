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
    
def _start_gray_ng_correction(self, gray:int, *, max_retries:int=3, thr_gamma=0.05, thr_c=0.003):
    """
    현재 _on_store에 방금 기록된 (white/main) gray 측정이 NG일 때,
    자코비안 g행만으로 Δh를 풀어 1회 보정→TV write→같은 gray 재측정.
    OK 되면 세션 재개, NG면 retry (최대 max_retries).
    """
    # 세션 일시정지
    self._pause_session(reason=f"gray={gray} NG")

    s = self._sess
    s['_gray_fix'] = {'g': int(gray), 'tries': 0, 'max': int(max_retries),
                      'thr_gamma': float(thr_gamma), 'thr_c': float(thr_c)}
    self._do_gray_fix_once()  # 첫 시도
    
def _do_gray_fix_once(self):
    ctx = self._sess.get('_gray_fix', None)
    if not ctx: 
        self._resume_session(); return
    g = ctx['g']; tries = ctx['tries']; maxr = ctx['max']
    thr_gamma = ctx['thr_gamma']; thr_c = ctx['thr_c']

    if tries >= maxr:
        logging.info(f"[GRAY-FIX] g={g} reached max retries → skip and resume")
        # 세션 재개: 다음 gray로 자연 진행되게끔 g_idx는 기존 루프가 제어
        self._sess['_gray_fix'] = None
        self._resume_session()
        return

    ctx['tries'] = tries + 1
    logging.info(f"[GRAY-FIX] g={g} try={ctx['tries']}/{maxr}")

    # ===== 1) Δ 타깃(해당 g만) =====
    def _get_off_on_xyG(store_off, store_on, gray):
        # xy/lv 추출
        tR = store_off['gamma']['main']['white'].get(gray, None)
        tO = store_on ['gamma']['main']['white'].get(gray, None)
        lv_r, cx_r, cy_r = (tR if tR else (np.nan, np.nan, np.nan))
        lv_o, cx_o, cy_o = (tO if tO else (np.nan, np.nan, np.nan))
        # 감마는 전체에서 계산 후 해당 g만 취함
        G_ref = self._compute_gamma_series(
            np.array([store_off['gamma']['main']['white'].get(i,(np.nan,)*3)[0] for i in range(256)], float)
        )
        G_on  = self._compute_gamma_series(
            np.array([store_on ['gamma']['main']['white'].get(i,(np.nan,)*3)[0] for i in range(256)], float)
        )
        return (G_on[gray]-G_ref[gray], cx_o-cx_r, cy_o-cy_r)

    dG, dCx, dCy = _get_off_on_xyG(self._off_store, self._on_store, g)

    # 소소한 deadband: 이미 충분히 작으면 바로 재측정으로 넘어가도 됨
    if (abs(dG) <= thr_gamma) and (abs(dCx) <= thr_c) and (abs(dCy) <= thr_c):
        logging.info(f"[GRAY-FIX] g={g} already within thr (skip fix) → remeasure")
        return self._remeasure_same_gray(g)

    # ===== 2) 자코비안 g행 구성 (결합 가중치는 기존과 동일)
    wG, wCx, wCy = 1.0, 0.05, 0.5  # 필요 시 UI/설정으로
    Ag = np.vstack([
        wG  * self.A_Gamma[g:g+1, :],   # (1,6K)
        wCx * self.A_Cx   [g:g+1, :],
        wCy * self.A_Cy   [g:g+1, :],
    ])                                  # (3,6K)
    b  = -np.array([wG*dG, wCx*dCx, wCy*dCy], dtype=np.float32)  # (3,)

    # ===== 3) 리지 해 구하기
    ATA = Ag.T @ Ag               # (6K,6K)
    rhs = Ag.T @ b               # (6K,)
    lambda_ridge = 1e-3
    ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
    delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)  # (6K,)

    # ===== 4) Δh → 256보정곡선으로 전개
    K   = len(self._jac_artifacts["knots"])
    Phi = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)

    idx=0
    dh_RL=delta_h[idx:idx+K]; idx+=K
    dh_GL=delta_h[idx:idx+K]; idx+=K
    dh_BL=delta_h[idx:idx+K]; idx+=K
    dh_RH=delta_h[idx:idx+K]; idx+=K
    dh_GH=delta_h[idx:idx+K]; idx+=K
    dh_BH=delta_h[idx:idx+K]

    corr = {
        "R_Low":  Phi @ dh_RL, "G_Low":  Phi @ dh_GL, "B_Low":  Phi @ dh_BL,
        "R_High": Phi @ dh_RH, "G_High": Phi @ dh_GH, "B_High": Phi @ dh_BH,
    }

    # ===== 5) 현재 TV LUT(캐시) → 4096→256 ↓ → 보정 적용
    vac_dict = self._vac_dict_cache
    lut256 = {
        "R_Low":  self._down4096_to_256(vac_dict["RchannelLow"]),
        "G_Low":  self._down4096_to_256(vac_dict["GchannelLow"]),
        "B_Low":  self._down4096_to_256(vac_dict["BchannelLow"]),
        "R_High": self._down4096_to_256(vac_dict["RchannelHigh"]),
        "G_High": self._down4096_to_256(vac_dict["GchannelHigh"]),
        "B_High": self._down4096_to_256(vac_dict["BchannelHigh"]),
    }
    lut256_new = {k: (lut256[k] + corr[k]).astype(np.float32) for k in lut256.keys()}

    # 안전 후처리(기존 파이프라인 재사용)
    for ch in ("R","G","B"):
        Lk, Hk = f"{ch}_Low", f"{ch}_High"
        # 엔드포인트 고정
        lut256_new[Lk][0]=0.0; lut256_new[Hk][0]=0.0
        lut256_new[Lk][255]=4095.0; lut256_new[Hk][255]=4095.0
        # 역전 방지→스무딩→mid nudge→최종 안전화
        low_fixed, high_fixed = self._fix_low_high_order(lut256_new[Lk], lut256_new[Hk])
        low_s  = self._smooth_and_monotone(low_fixed, 9)
        high_s = self._smooth_and_monotone(high_fixed, 9)
        low_m, high_m = self._nudge_midpoint(low_s, high_s, max_err=3.0, strength=0.5)
        lut256_new[Lk], lut256_new[Hk] = self._finalize_channel_pair_safely(low_m, high_m)

    # ===== 6) 256→4096 ↑, JSON 구성, TV write → read → 같은 gray 재측정
    new_lut_4096 = {
        "RchannelLow":  self._up256_to_4096(lut256_new["R_Low"]),
        "GchannelLow":  self._up256_to_4096(lut256_new["G_Low"]),
        "BchannelLow":  self._up256_to_4096(lut256_new["B_Low"]),
        "RchannelHigh": self._up256_to_4096(lut256_new["R_High"]),
        "GchannelHigh": self._up256_to_4096(lut256_new["G_High"]),
        "BchannelHigh": self._up256_to_4096(lut256_new["B_High"]),
    }
    for k in new_lut_4096:
        new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

    vac_write_json = self.build_vacparam_std_format(
        base_vac_dict=self._vac_dict_cache,
        new_lut_tvkeys=new_lut_4096
    )

    def _after_write(ok, msg):
        logging.info(f"[GRAY-FIX] write: {ok} {msg}")
        if not ok:
            return self._remeasure_same_gray(g)  # 일단 재측정 시도 후 판단

        self._read_vac_from_tv(lambda vd: self._after_fix_read_and_remeasure(vd, g))

    self._write_vac_to_tv(vac_write_json, on_finished=_after_write)
    
def _after_fix_read_and_remeasure(self, vac_dict_after, gray:int):
    if vac_dict_after:
        self._vac_dict_cache = vac_dict_after
    self._remeasure_same_gray(gray)

def _remeasure_same_gray(self, gray:int):
    """같은 g를 즉시 재측정한다(white/main만)."""
    # 차트/테이블에서 ON 시리즈에 덮어쓰도록 그대로 측정 루틴 재사용
    # 단, 세션은 여전히 paused 상태. g_idx는 증가시키지 않음.
    self.changeColor(f"{gray},{gray},{gray}")
    # settle 후 한 페어 측정만 트리거
    def done_pair(pattern, g):
        # 기존 핸들러를 재사용하되, g_idx 증가는 막아야 함 (아래 4번 패치 참고)
        self._trigger_gamma_pair(pattern='white', gray=g)
    QTimer.singleShot(self._sess.get('cs_settle_ms', 1000), lambda: done_pair('white', gray))
    