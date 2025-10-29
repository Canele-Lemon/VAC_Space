def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
    logging.info(f"[CORR] iteration {iter_idx} start")
    self._step_start(2)

    # 1) 현재 TV LUT (캐시) 확보
    if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
        logging.warning("[CORR] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
        return None
    vac_dict = self._vac_dict_cache  # TV에서 읽어온 최신 VAC JSON (4096포인트, 12bit)

    # 2) 4096 → 256 다운샘플 (Low/High 전채널)
    vac_lut_4096 = {
        "R_Low":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
        "R_High": np.asarray(vac_dict["RchannelHigh"], dtype=np.float32),
        "G_Low":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
        "G_High": np.asarray(vac_dict["GchannelHigh"], dtype=np.float32),
        "B_Low":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
        "B_High": np.asarray(vac_dict["BchannelHigh"], dtype=np.float32),
    }

    lut256 = {
        "R_Low":  self._down4096_to_256(vac_lut_4096["R_Low"]),
        "G_Low":  self._down4096_to_256(vac_lut_4096["G_Low"]),
        "B_Low":  self._down4096_to_256(vac_lut_4096["B_Low"]),
        "R_High": self._down4096_to_256(vac_lut_4096["R_High"]),
        "G_High": self._down4096_to_256(vac_lut_4096["G_High"]),
        "B_High": self._down4096_to_256(vac_lut_4096["B_High"]),
    }
    # lut256[...] 은 여전히 0~4095 스케일 (12bit 값) 상태입니다.
    lut256_before = {k: v.copy() for k, v in lut256.items()}

    # 3) Δ 목표(white/main 기준): OFF vs ON 차이
    #    Gamma: 1..254 유효, Cx/Cy: 0..255
    d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
    # d_targets = {"Gamma":(256,), "Cx":(256,), "Cy":(256,)}, 값 = (ON - OFF)

    # 아주 작은 오차(이미 충분히 맞은 gray)는 굳이 고치지 말자 → 0으로
    thr_c = 0.003
    thr_gamma = 0.03
    for g in range(256):
        if (
            abs(d_targets["Cx"][g]) <= thr_c and
            abs(d_targets["Cy"][g]) <= thr_c and
            abs(d_targets["Gamma"][g]) <= thr_gamma
        ):
            d_targets["Cx"][g]    = 0.0
            d_targets["Cy"][g]    = 0.0
            d_targets["Gamma"][g] = 0.0

    # 4) 결합 선형계
    #    ΔY ≈ [A_Gamma; A_Cx; A_Cy] · Δh
    #    여기서 A_* shape = (256, 6K). Δh shape = (6K,)
    #    wG, wCx, wCy는 가중치
    wCx = 0.05
    wCy = 0.5
    wG  = 1.0

    A_cat = np.vstack([
        wG  * self.A_Gamma,
        wCx * self.A_Cx,
        wCy * self.A_Cy
    ]).astype(np.float32)  # (256*3, 6K)

    b_cat = -np.concatenate([
        wG  * d_targets["Gamma"],
        wCx * d_targets["Cx"],
        wCy * d_targets["Cy"]
    ]).astype(np.float32)  # (256*3,)

    # 유효치 마스크(특히 gamma의 NaN 등에서 온 0/inf 제거)
    mask = np.isfinite(b_cat)
    A_use = A_cat[mask, :]  # (M, 6K)
    b_use = b_cat[mask]     # (M,)

    # 5) 리지 회귀 해 (Δh) 구하기
    #    (A^T A + λI) Δh = A^T b
    ATA = A_use.T @ A_use            # (6K,6K)
    rhs = A_use.T @ b_use            # (6K,)
    ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
    delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)  # (6K,)

    # --------- 디버그 준비 -------------
    K = len(self._jac_artifacts["knots"])
    knots = np.asarray(self._jac_artifacts["knots"], dtype=np.int32)

    # 6) knot delta → per-gray 보정곡선(256포인트)로 전개
    #    delta_h 해석:
    #    [R_Low_knots(0:K),
    #     G_Low_knots(K:2K),
    #     B_Low_knots(2K:3K),
    #     R_High_knots(3K:4K),
    #     G_High_knots(4K:5K),
    #     B_High_knots(5K:6K)]
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

    # 7) 1차 LUT 후보 (아직 후처리 전)
    lut256_new = {
        "R_Low":  (lut256["R_Low"]  + corr_RL).astype(np.float32),
        "G_Low":  (lut256["G_Low"]  + corr_GL).astype(np.float32),
        "B_Low":  (lut256["B_Low"]  + corr_BL).astype(np.float32),
        "R_High": (lut256["R_High"] + corr_RH).astype(np.float32),
        "G_High": (lut256["G_High"] + corr_GH).astype(np.float32),
        "B_High": (lut256["B_High"] + corr_BH).astype(np.float32),
    }

    # =========================
    # ▼ NEW: 안전 후처리 파이프라인
    # =========================
    #
    # 목적:
    #   - 톱니/지글지글 완화 (moving average + monotone)
    #   - Low > High 금지
    #   - (Low+High)/2 가 이상하게 튀는 gray에서만 살짝 눌러서
    #     감마/휘도 급튜는 구간 줄이기
    #   - g=0은 항상 0, g=255는 항상 4095
    #   - 최종적으로 다시 monotone + clip

    for ch in ("R", "G", "B"):
        Lk = f"{ch}_Low"
        Hk = f"{ch}_High"

        # (0) 엔드포인트를 미리 합리적으로 잡아준다.
        lut256_new[Lk][0]   = 0.0
        lut256_new[Hk][0]   = 0.0
        lut256_new[Lk][255] = 4095.0
        lut256_new[Hk][255] = 4095.0

        # (1) Low/High 역전 금지 (1차 정리)
        low_fixed, high_fixed = self._fix_low_high_order(
            lut256_new[Lk], lut256_new[Hk]
        )

        # (2) 스무딩 + 단조 증가 보장으로 톱니 제거
        low_smooth  = self._smooth_and_monotone(low_fixed,  win=9)
        high_smooth = self._smooth_and_monotone(high_fixed, win=9)

        # (3) 평균 밝기(mid) 너무 이상하게 튀는 지점만 살짝 눌러주기
        low_mid, high_mid = self._nudge_midpoint(
            low_smooth, high_smooth,
            max_err=3.0,    # 12bit에서 ±3카운트 이상 벗어나면만 관여
            strength=0.5    # 그 오차의 절반만 교정
        )

        # (4) 최종 안전화:
        #     - 다시 Low<=High
        #     - 단조 재보장
        #     - 0/255 엔드포인트 고정
        #     - clip(0..4095)
        low_final, high_final = self._finalize_channel_pair_safely(
            low_mid, high_mid
        )

        lut256_new[Lk] = low_final
        lut256_new[Hk] = high_final

    # 이제 lut256_new[*] 는
    #  - 단조 증가
    #  - Low <= High
    #  - g=0 → 0, g=255 → 4095
    #  - 고주파 톱니 줄어듦
    # =========================
    # ▲ NEW 파이프라인 끝
    # =========================

    # --------- 디버그 로그 (보정량 요약) -------------
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
    # -------------------------------------------------

    # 8) 256 → 4096 업샘플 (모든 채널), 정수화
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

    # 9) UI용 차트/테이블 갱신
    lut_dict_plot = {
        "R_Low":  new_lut_4096["RchannelLow"],
        "R_High": new_lut_4096["RchannelHigh"],
        "G_Low":  new_lut_4096["GchannelLow"],
        "G_High": new_lut_4096["GchannelHigh"],
        "B_Low":  new_lut_4096["BchannelLow"],
        "B_High": new_lut_4096["BchannelHigh"],
    }
    self._update_lut_chart_and_table(lut_dict_plot)
    self._step_done(2)

    # 10) TV에 쓰고, 다시 읽고, 다시 측정 → 스펙 체크 흐름은 기존 그대로
    def _after_write(ok, msg):
        logging.info(f"[VAC Write] {msg}")
        if not ok:
            return
        # 쓰기 성공 → 재읽기
        logging.info("보정 LUT TV Reading 시작")
        self._read_vac_from_tv(_after_read_back)

    def _after_read_back(vac_dict_after):
        if not vac_dict_after:
            logging.error("보정 후 VAC 재읽기 실패")
            return
        logging.info("보정 LUT TV Reading 완료")
        self._step_done(3)

        # 1) 캐시/차트 갱신
        self._vac_dict_cache = vac_dict_after

        # 2) ON 시리즈 리셋 (OFF는 참조 유지)
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        # 3) 보정 후(=ON) 측정 세션 시작
        profile_corr = SessionProfile(
            legend_text=f"CORR #{iter_idx}",
            cie_label=None,
            table_cols={
                "lv":4, "cx":5, "cy":6, "gamma":7,
                "d_cx":8, "d_cy":9, "d_gamma":10
            },
            ref_store=self._off_store  # 항상 OFF 대비 Δ
        )

        def _after_corr(store_corr):
            self._step_done(4)
            self._on_store = store_corr  # 최신 ON(보정 후) 측정 결과

            self._step_start(5)
            self._spec_thread = SpecEvalThread(
                self._off_store, self._on_store,
                thr_gamma=0.05, thr_c=0.003, parent=self
            )
            self._spec_thread.finished.connect(
                lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx, max_iters)
            )
            self._spec_thread.start()

        logging.info("보정 LUT 기준 측정 시작")
        self._step_start(4)
        self.start_viewing_angle_session(
            profile=profile_corr,
            gray_levels=getattr(op, "gray_levels_256", list(range(256))),
            gamma_patterns=('white',),             # white만 측정
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            cs_settle_ms=1000,
            on_done=_after_corr
        )

    logging.info("LUT {}차 보정 완료".format(iter_idx))
    logging.info("LUT {}차 TV Writing 시작".format(iter_idx))

    # 11) VAC JSON 재조립 후 TV에 write
    vac_write_json = self.build_vacparam_std_format(
        base_vac_dict=self._vac_dict_cache,
        new_lut_tvkeys=new_lut_4096
    )
    self._write_vac_to_tv(vac_write_json, on_finished=_after_write)