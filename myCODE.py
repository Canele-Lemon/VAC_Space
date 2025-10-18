def build_vacparam_std_format(self, vac_cache_std: dict, lut_std: dict) -> str:
    """
    vac_cache_std : TV에서 읽어온 '원래 키' JSON dict (제어필드 포함)  ex) DRV_valc_..., RchannelLow...
    lut_std       : {'R_Low','R_High','G_Low','G_High','B_Low','B_High'}  각각 4096 길이
    return        : TV에 쓰기용 JSON 문자열 (원래 키 유지)
    """
    try:
        # 0) LUT 길이 확인
        req = ["R_Low","R_High","G_Low","G_High","B_Low","B_High"]
        for k in req:
            if k not in lut_std or len(lut_std[k]) != 4096:
                raise ValueError(f"[build_vacparam_std_format] invalid {k} length={len(lut_std.get(k, []))}")

        # 1) 제어필드 유지 + LUT만 교체
        out = dict(vac_cache_std)  # shallow copy

        # 원래 JSON 키로 매핑 (channel 이름 유지)
        out["RchannelLow"]  = list(map(int, lut_std["R_Low"]))
        out["RchannelHigh"] = list(map(int, lut_std["R_High"]))
        out["GchannelLow"]  = list(map(int, lut_std["G_Low"]))
        out["GchannelHigh"] = list(map(int, lut_std["G_High"]))
        out["BchannelLow"]  = list(map(int, lut_std["B_Low"]))
        out["BchannelHigh"] = list(map(int, lut_std["B_High"]))

        # 2) JSON 직렬화 (최소 공백, 한 줄)
        return json.dumps(out, separators=(",", ":"))
    except Exception as e:
        logging.exception(e)
        return None
        
        
def _apply_vac_from_db_and_measure_on(self):
    # (A) VAC ON 보장 (혹시 OFF라면 시도, 실패 시 종료)
    st = self.check_VAC_status()
    if not st.get("activated", False):
        logging.info("[VAC] 현재 OFF → ON 전환 시도")
        if not self._set_vac_active(True):
            logging.error("[VAC] ON 전환 실패 — 최적화 종료")
            return

    # (B) DB에서 Panel_Maker + Frame_Rate 조합 VAC_Data 가져오기
    panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
    fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
    vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
    if vac_data is None:
        logging.error(f"[VAC] {panel}+{fr} 매칭 VAC Data 없음 — 종료")
        return

    # (C) TV에 VAC 쓰기 → 읽기 → LUT/테이블 갱신 → ON 측정 세션 시작
    def _after_write(ok, msg):
        logging.info(f"[VAC Write] {msg}")
        if not ok:
            logging.error("[VAC] 쓰기 실패 — 종료")
            return
        self._read_vac_from_tv(lambda vac_dict: _after_read(vac_dict))  # 완료 콜백

    def _after_read(vac_dict):
        if not vac_dict:
            logging.error("[VAC] 읽기 실패 — 종료")
            return

        # 1) TV 원본 JSON 그 상태 그대로 캐시(제어필드 포함)
        self._vac_cache_std = vac_dict

        # 2) 차트/테이블 갱신용 키로 변환(이 화면에서만 쓸 임시 dict)
        #    "RchannelLow" → "R_Low" 처럼 키 이름만 치환해서 넘김
        vac_lut_dict = {}
        for k, v in vac_dict.items():
            if k.endswith("channelLow"):
                vac_lut_dict[k[0] + "_Low"] = v   # 'RchannelLow' -> 'R_Low'
            elif k.endswith("channelHigh"):
                vac_lut_dict[k[0] + "_High"] = v  # 'GchannelHigh' -> 'G_High'
        self._update_lut_chart_and_table(vac_lut_dict)

        # 3) VAC ON 측정 세션
        gamma_lines_on = {
            'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"VAC ON - {p}")
                     for p in ('white', 'red', 'green', 'blue')},
            'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=f"VAC ON - {p}")
                     for p in ('white', 'red', 'green', 'blue')},
        }
        profile_on = SessionProfile(
            legend_text="VAC ON",
            cie_label="data_2",
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

        def _after_on(store_on):
            self._on_store = store_on
            if self._check_spec_pass(self._off_store, self._on_store):
                logging.info("✅ 스펙 통과 — 종료")
                return
            # (D) 반복 보정 시작
            self._run_correction_iteration(iter_idx=1)

        self.start_viewing_angle_session(
            profile=profile_on, gamma_lines=gamma_lines_on,
            gray_levels=getattr(op, "gray_levels_256", list(range(256))),
            patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_on
        )

    # (D) 쓰기 시작
    self._write_vac_to_tv(vac_data, on_finished=_after_write)
    
def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
    logging.info(f"[CORR] iteration {iter_idx} start")

    # 1) 현재 TV LUT : 캐시에서 꺼냄(원래키)
    if not hasattr(self, "_vac_cache_std") or not self._vac_cache_std:
        logging.error("[CORR] VAC 캐시 없음 — 직전 읽기가 필요합니다.")
        return
    vac_src = self._vac_cache_std

    # 2) 4096→256 다운샘플 (High만 수정, Low 고정)
    #    원래 키 → 표준 LUT 키로 꺼내 계산
    lut_cur_std = {
        "R_Low":  np.asarray(vac_src["RchannelLow"],  dtype=np.float32),
        "R_High": np.asarray(vac_src["RchannelHigh"], dtype=np.float32),
        "G_Low":  np.asarray(vac_src["GchannelLow"],  dtype=np.float32),
        "G_High": np.asarray(vac_src["GchannelHigh"], dtype=np.float32),
        "B_Low":  np.asarray(vac_src["BchannelLow"],  dtype=np.float32),
        "B_High": np.asarray(vac_src["BchannelHigh"], dtype=np.float32),
    }

    high_256 = {ch: self._down4096_to_256(lut_cur_std[ch]) for ch in ['R_High','G_High','B_High']}
    # low_256 = {ch: self._down4096_to_256(lut_cur_std[ch]) for ch in ['R_Low','G_Low','B_Low']}  # 계산엔 필요없지만 참고용

    # 3) Δ 목표(white/main 기준)
    d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)

    # 4) 결합 선형계 : [wG*Aγ; wC*A_Cx; wC*A_Cy] Δh = - [wG*Δγ; wC*ΔCx; wC*ΔCy]
    wG, wC = 1.0, 1.0
    A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
    b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)

    mask = np.isfinite(b_cat)
    A_use = A_cat[mask, :]
    b_use = b_cat[mask]

    ATA = A_use.T @ A_use
    rhs = A_use.T @ b_use
    ATA[np.diag_indices_from(ATA)] += lambda_ridge
    delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

    # 5) Δcurve = Phi * Δh_channel → High 256 보정
    K    = len(self._jac_artifacts["knots"])
    dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
    Phi  = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)
    corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

    high_256_new = {
        "R_High": (high_256["R_High"] + corr_R).astype(np.float32),
        "G_High": (high_256["G_High"] + corr_G).astype(np.float32),
        "B_High": (high_256["B_High"] + corr_B).astype(np.float32),
    }

    # 6) 단조/클립 → 12bit 업샘플
    for ch in high_256_new:
        self._enforce_monotone(high_256_new[ch])
        high_256_new[ch] = np.clip(high_256_new[ch], 0, 4095)

    lut_new_std = {
        "R_Low":  lut_cur_std["R_Low"].copy(),  # Low는 유지
        "G_Low":  lut_cur_std["G_Low"].copy(),
        "B_Low":  lut_cur_std["B_Low"].copy(),
        "R_High": self._up256_to_4096(high_256_new["R_High"]),
        "G_High": self._up256_to_4096(high_256_new["G_High"]),
        "B_High": self._up256_to_4096(high_256_new["B_High"]),
    }

    # 7) TV 쓰기용 JSON 구성(제어필드는 캐시 유지)
    vac_json_write = self.build_vacparam_std_format(self._vac_cache_std, lut_new_std)
    if not vac_json_write:
        logging.error("[CORR] VAC JSON build 실패 — 종료")
        return

    # 8) TV에 적용 → 읽기 → 차트 갱신
    def _after_write(ok, msg):
        logging.info(f"[CORR Write] {msg}")
        if not ok:
            logging.error("[CORR] 쓰기 실패 — 종료")
            return
        self._read_vac_from_tv(lambda vac_dict: _after_read(vac_dict))

    def _after_read(vac_dict):
        if not vac_dict:
            logging.error("[CORR] 읽기 실패 — 종료")
            return

        # 최신 캐시로 교체
        self._vac_cache_std = vac_dict

        # 차트용 키 변환해서 갱신
        vac_lut_dict = {}
        for k, v in vac_dict.items():
            if k.endswith("channelLow"):
                vac_lut_dict[k[0] + "_Low"] = v
            elif k.endswith("channelHigh"):
                vac_lut_dict[k[0] + "_High"] = v
        self._update_lut_chart_and_table(vac_lut_dict)

        # 9) 재측정 (CORR i)
        label = f"VAC ON (CORR{iter_idx})"
        gamma_lines_corr = {
            'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"{label} - {p}")
                     for p in ('white','red','green','blue')},
            'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=f"{label} - {p}")
                     for p in ('white','red','green','blue')},
        }
        profile_corr = SessionProfile(
            legend_text=label,
            cie_label=f"data_{iter_idx+2}",
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

        def _after_corr(store_corr):
            self._on_store = store_corr  # 최신으로 갱신
            if self._check_spec_pass(self._off_store, self._on_store):
                logging.info("✅ 스펙 통과 — 종료")
                return
            if iter_idx >= max_iters:
                logging.info("ℹ️ 최대 보정 횟수 도달 — 종료")
                return
            # 다음 라운드
            self._run_correction_iteration(iter_idx+1, max_iters=max_iters)

        self.start_viewing_angle_session(
            profile=profile_corr, gamma_lines=gamma_lines_corr,
            gray_levels=getattr(op, "gray_levels_256", list(range(256))),
            patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_corr
        )

    self._write_vac_to_tv(vac_json_write, on_finished=_after_write)