def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
    logging.info(f"[CORR] iteration {iter_idx} start")
    self._step_start(2)

    # ... (1) LUT 읽어서 lut256 만들고 etc. 기존 그대로 ...

    # 2) 4096 → 256 다운샘플
    vac_dict = self._vac_dict_cache
    lut256 = {
        "R_Low":  self._down4096_to_256(vac_dict["RchannelLow"]).astype(np.float32),
        "G_Low":  self._down4096_to_256(vac_dict["GchannelLow"]).astype(np.float32),
        "B_Low":  self._down4096_to_256(vac_dict["BchannelLow"]).astype(np.float32),
        "R_High": self._down4096_to_256(vac_dict["RchannelHigh"]).astype(np.float32),
        "G_High": self._down4096_to_256(vac_dict["GchannelHigh"]).astype(np.float32),
        "B_High": self._down4096_to_256(vac_dict["BchannelHigh"]).astype(np.float32),
    }

    # ▼ 앞으로 바뀔 애들 대비해서 “보정 전 LUT256 상태”를 복사해 둡니다.
    lut256_before = {k: v.copy() for k, v in lut256.items()}

    # ... (3) d_targets 만들고 etc. 기존 그대로 ...

    # 4) 선형계 구축
    wG, wC = 1.0, 1.0  # 혹은 당신이 지금 실험 중인 값
    A_cat = np.vstack([
        wG * self.A_Gamma,
        wC * self.A_Cx,
        wC * self.A_Cy
    ]).astype(np.float32)
    b_cat = -np.concatenate([
        wG * d_targets["Gamma"],
        wC * d_targets["Cx"],
        wC * d_targets["Cy"]
    ]).astype(np.float32)

    mask = np.isfinite(b_cat)
    A_use = A_cat[mask, :]
    b_use = b_cat[mask]

    # 5) ridge solve -> delta_h
    ATA = A_use.T @ A_use
    rhs = A_use.T @ b_use
    ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
    delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)   # (6K,)

    # --------- NEW: knot 변화 디버그 로그 (solve 직후) -------------
    K = len(self._jac_artifacts["knots"])
    knots = np.asarray(self._jac_artifacts["knots"], dtype=np.int32)

    # corr_* 계산 전이지만, delta_h만으로 어떤 knot(=제어점)가 얼마나 움직이려 하는지 볼 수 있음
    # 다만 사용자 눈에는 실제 LUT 변화(Low/High 등)도 궁금하니,
    # 일단 여기서는 delta_h만 먼저 저장해 두고,
    # lut256_after를 만든 뒤에 한 번에 찍을게요.
    # --------------------------------------------------------------

    # 6) knot delta -> per-gray correction
    Phi = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)

    idx0 = 0
    dh_RL = delta_h[idx0      : idx0+K]; idx0 += K
    dh_GL = delta_h[idx0      : idx0+K]; idx0 += K
    dh_BL = delta_h[idx0      : idx0+K]; idx0 += K
    dh_RH = delta_h[idx0      : idx0+K]; idx0 += K
    dh_GH = delta_h[idx0      : idx0+K]; idx0 += K
    dh_BH = delta_h[idx0      : idx0+K]

    corr_RL = Phi @ dh_RL  # (256,)
    corr_GL = Phi @ dh_GL
    corr_BL = Phi @ dh_BL
    corr_RH = Phi @ dh_RH
    corr_GH = Phi @ dh_GH
    corr_BH = Phi @ dh_BH

    # 7) LUT256 갱신 (12bit 그대로)
    lut256_new = {
        "R_Low":  (lut256["R_Low"]  + corr_RL).astype(np.float32),
        "G_Low":  (lut256["G_Low"]  + corr_GL).astype(np.float32),
        "B_Low":  (lut256["B_Low"]  + corr_BL).astype(np.float32),
        "R_High": (lut256["R_High"] + corr_RH).astype(np.float32),
        "G_High": (lut256["G_High"] + corr_GH).astype(np.float32),
        "B_High": (lut256["B_High"] + corr_BH).astype(np.float32),
    }

    # monotone + clip
    for ch in lut256_new:
        self._enforce_monotone(lut256_new[ch])
        lut256_new[ch] = np.clip(lut256_new[ch], 0, 4095)

    # --------- NEW: 여기서 디버그 로그 호출 -------------
    try:
        self._debug_log_knot_update(
            iter_idx=iter_idx,
            knots=knots,
            delta_h=delta_h,
            lut256_before=lut256_before,
            lut256_after=lut256_new,
        )
    except Exception:
        logging.exception("[CORR DEBUG] _debug_log_knot_update failed")
    # ----------------------------------------------------

    # 8) 256 -> 4096 업샘플해서 TV write ... 이하 기존 그대로
    new_lut_4096 = {
        "RchannelLow":  self._up256_to_4096(lut256_new["R_Low"]),
        "GchannelLow":  self._up256_to_4096(lut256_new["G_Low"]),
        "BchannelLow":  self._up256_to_4096(lut256_new["B_Low"]),
        "RchannelHigh": self._up256_to_4096(lut256_new["R_High"]),
        "GchannelHigh": self._up256_to_4096(lut256_new["G_High"]),
        "BchannelHigh": self._up256_to_4096(lut256_new["B_High"]),
    }
    # ... 나머지 clip/round, TV write, spec check 등 기존 동일 ...