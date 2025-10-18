# ===== [NEW] 표준 LUT ↔ TV JSON 키 변환 =====
def _tvjson_to_std_lut(self, vac_dict: dict) -> dict:
    """TV JSON 키(RchannelLow...) → 표준 키(R_Low...)로 변환"""
    keymap = {
        "RchannelLow":  "R_Low",
        "RchannelHigh": "R_High",
        "GchannelLow":  "G_Low",
        "GchannelHigh": "G_High",
        "BchannelLow":  "B_Low",
        "BchannelHigh": "B_High",
    }
    out = {}
    for src, dst in keymap.items():
        if src not in vac_dict:
            raise KeyError(f"[tv→std] missing key: {src}")
        out[dst] = vac_dict[src]
    return out

def _std_lut_to_tvjson(self, std_lut: dict) -> dict:
    """표준 키(R_Low...) → TV JSON 키(RchannelLow...)로 변환 (LUT 부분만)"""
    keymap = {
        "R_Low":  "RchannelLow",
        "R_High": "RchannelHigh",
        "G_Low":  "GchannelLow",
        "G_High": "GchannelHigh",
        "B_Low":  "BchannelLow",
        "B_High": "BchannelHigh",
    }
    out = {}
    for src, dst in keymap.items():
        if src not in std_lut:
            raise KeyError(f"[std→tv] missing key: {src}")
        out[dst] = std_lut[src]
    return out


# ===== [NEW] 제어 필드만 추출/유틸 =====
def _extract_control_fields(self, vacparam_dict: dict) -> dict:
    ctrl_keys = [
        "DRV_valc_major_ctrl",
        "DRV_valc_pattern_ctrl_0",
        "DRV_valc_pattern_ctrl_1",
        "DRV_valc_sat_ctrl",
        "DRV_valc_hpf_ctrl_0",
        "DRV_valc_hpf_ctrl_1",
    ]
    ctrl = {}
    for k in ctrl_keys:
        if k not in vacparam_dict:
            raise KeyError(f"[VAC build] control key missing: {k}")
        ctrl[k] = vacparam_dict[k]
    return ctrl

def _coerce_lut_4096(self, arr_like) -> list:
    import numpy as np
    arr = np.asarray(arr_like, dtype=np.float64).copy()
    if arr.size < 4096:
        pad_val = arr[-1] if arr.size > 0 else 0
        arr = np.pad(arr, (0, 4096 - arr.size), mode='constant', constant_values=pad_val)
    elif arr.size > 4096:
        x = np.linspace(0, 1, arr.size)
        xq = np.linspace(0, 1, 4096)
        arr = np.interp(xq, x, arr)
    arr = np.clip(np.rint(arr), 0, 4095).astype(np.int32)
    return arr.tolist()


# ===== [NEW/REPLACE] 표준 LUT + 기존 제어필드 → 최종 JSON 문자열 =====
def build_vacparam_std_format(self, lut_new_std: dict, base_ctrl: dict = None) -> str:
    """
    입력: lut_new_std = {R_Low, R_High, G_Low, G_High, B_Low, B_High} 각 4096
    제어필드: base_ctrl(dict) 없으면 self._vac_cache_std(최근 TV JSON)에서 추출
    출력: 장치에 바로 쓸 수 있는 JSON 문자열
    """
    if base_ctrl is None:
        if not hasattr(self, "_vac_cache_std") or not isinstance(self._vac_cache_std, dict):
            raise RuntimeError("[VAC build] no control source (_vac_cache_std is empty).")
        ctrl_src = self._vac_cache_std
    else:
        ctrl_src = base_ctrl

    controls = self._extract_control_fields(ctrl_src)

    # LUT(표준→TV키) + 4096 강제
    lut_tv = self._std_lut_to_tvjson(lut_new_std)
    for k in lut_tv:
        lut_tv[k] = self._coerce_lut_4096(lut_tv[k])

    vacparam_out = {**controls, **lut_tv}
    return json.dumps(vacparam_out, ensure_ascii=False, separators=(',', ':'))


# ===== [FIX] VAC ON/OFF 토글 유틸 (소문자 true/false) =====
def _set_vac_active(self, enable: bool) -> bool:
    try:
        self.send_command(self.ser_tv, 's')
        cmd = (
            "luna-send -n 1 -f "
            "luna://com.webos.service.panelcontroller/setVACActive "
            f"'{{\"OnOff\":{str(enable).lower()}}}'"
        )
        self.send_command(self.ser_tv, cmd)
        self.send_command(self.ser_tv, 'exit')
        time.sleep(0.5)
        st = self.check_VAC_status()
        return bool(st.get("activated", False)) == enable
    except Exception as e:
        logging.error(f"VAC {'ON' if enable else 'OFF'} 전환 실패: {e}")
        return False


# ===== [REPLACE] _run_off_baseline_then_on() 내부 VAC ON 전환 부분 고치기 =====
def _run_off_baseline_then_on(self):
    gamma_lines_off = {
        'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"VAC OFF (Ref.) - {p}")
                 for p in ('white','red','green','blue')},
        'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=f"VAC OFF (Ref.) - {p}")
                 for p in ('white','red','green','blue')},
    }
    profile_off = SessionProfile(
        legend_text="VAC OFF (Ref.)",
        cie_label="data_1",
        table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
        ref_store=None
    )

    def _after_off(store_off):
        self._off_store = store_off

        st = self.check_VAC_status()
        if not st.get("activated", False):
            logging.info("현재 VAC OFF 상태입니다. VAC ON 전환 시도...")
            ok = self._set_vac_active(True)
            if ok:
                logging.info("VAC ON 전환 성공")
            else:
                logging.warning("VAC ON 전환 실패로 보입니다. 그래도 VAC 데이터 적용 진행")
        else:
            logging.debug("이미 VAC ON 상태. VAC 데이터 적용으로 바로 진행")

        self._apply_vac_from_db_and_measure_on()

    self.start_viewing_angle_session(
        profile=profile_off, gamma_lines=gamma_lines_off,
        gray_levels=list(range(256)), patterns=('white','red','green','blue'),
        colorshift_patterns=op.colorshift_patterns,
        first_gray_delay_ms=3000, cs_settle_ms=1000,
        on_done=_after_off
    )


# ===== [REPLACE] _apply_vac_from_db_and_measure_on() : 읽기 후 캐시/변환 =====
def _apply_vac_from_db_and_measure_on(self):
    panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
    fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
    vac_pk, vac_version, vac_data_json = self._fetch_vac_by_model(panel, fr)
    if vac_data_json is None:
        logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다.")
        return

    def _after_write(ok, msg):
        logging.info(f"[VAC Write] {msg}")
        if not ok:
            return
        self._read_vac_from_tv(lambda vac: _after_read(vac))

    def _after_read(vac_dict):
        if vac_dict:
            # 1) TV원본 전체 캐시(제어필드 포함)
            self._vac_cache_std = vac_dict
            # 2) 표준 LUT 캐시(보정루틴용)
            try:
                self._lut_cache_std = self._tvjson_to_std_lut(vac_dict)
            except Exception as e:
                logging.exception(e)
                self._lut_cache_std = None

            # 3) 차트/테이블 업데이트는 표준키로
            if self._lut_cache_std:
                self._update_lut_chart_and_table(self._lut_cache_std)

        # VAC ON 측정 세션 시작
        gamma_lines_on = {
            'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label="VAC ON")
                     for p in ('white','red','green','blue')},
            'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label="VAC ON")
                     for p in ('white','red','green','blue')},
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
                logging.info("축하합니다! 스펙 통과 — 종료")
                return
            # 미통과 → 보정 사이클
            self._run_one_correction_cycle(iter_idx=1)

        self.start_viewing_angle_session(
            profile=profile_on, gamma_lines=gamma_lines_on,
            gray_levels=op.gray_levels, patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_on
        )

    # DB JSON 그대로 쓰기 (장비 포맷)
    self._write_vac_to_tv(vac_data_json, on_finished=_after_write)


# ===== [NEW] 보정 사이클 오케스트레이터: 계산 → JSON → 쓰기 → 재측정 =====
def _run_one_correction_cycle(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
    lut_new_std = self._run_correction_iteration(iter_idx=iter_idx,
                                                 max_iters=max_iters,
                                                 lambda_ridge=lambda_ridge)
    if lut_new_std is None:
        logging.error("[CORR] 보정 LUT 계산 실패")
        return

    vac_json = self.build_vacparam_std_format(lut_new_std)  # 제어필드는 캐시에서 자동 사용

    def _after_write(ok, msg):
        logging.info(f"[CORR WRITE] {msg}")
        if not ok:
            return

        def _after_read(vac_dict):
            # 캐시 최신화
            if vac_dict:
                self._vac_cache_std = vac_dict
                try:
                    self._lut_cache_std = self._tvjson_to_std_lut(vac_dict)
                except Exception as e:
                    logging.exception(e)
                    self._lut_cache_std = None
                if self._lut_cache_std:
                    self._update_lut_chart_and_table(self._lut_cache_std)

            # 재측정 세션(CORR i)
            label = f"VAC ON (CORR{iter_idx})"
            gamma_lines_corr = {
                'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=label)
                         for p in ('white','red','green','blue')},
                'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=label)
                         for p in ('white','red','green','blue')},
            }
            profile_corr = SessionProfile(
                legend_text=label,
                cie_label=f"data_{iter_idx+2}",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._on_store = store_corr
                if self._check_spec_pass(self._off_store, self._on_store):
                    logging.info("스펙 통과 — 종료")
                    return
                if iter_idx >= max_iters:
                    logging.info("최대 보정 횟수 도달 — 종료")
                    return
                self._run_one_correction_cycle(iter_idx+1, max_iters=max_iters, lambda_ridge=lambda_ridge)

            self.start_viewing_angle_session(
                profile=profile_corr, gamma_lines=gamma_lines_corr,
                gray_levels=list(range(256)), patterns=('white','red','green','blue'),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000, cs_settle_ms=1000,
                on_done=_after_corr
            )

        self._read_vac_from_tv(_after_read)

    self._write_vac_to_tv(vac_json, on_finished=_after_write)


# ===== [REPLACE] _run_correction_iteration(): 보정 LUT 계산만 하고 반환 =====
def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
    logging.info(f"[CORR] iteration {iter_idx} start")

    # 1) 현재 LUT(표준 캐시) 확보
    if not hasattr(self, "_lut_cache_std") or self._lut_cache_std is None:
        logging.warning("[CORR] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
        return None
    lut_cur = self._lut_cache_std  # 표준 키 dict

    # 2) 4096→256 (High만 수정)
    high_256 = {ch: self._down4096_to_256(lut_cur[ch]) for ch in ['R_High','G_High','B_High']}
    # low_256  = {ch: self._down4096_to_256(lut_cur[ch]) for ch in ['R_Low','G_Low','B_Low']} # 고정이므로 미사용

    # 3) Δ 목표
    d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)

    # 4) 자코비안 결합
    wG, wC = 1.0, 1.0
    A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
    b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)

    mask = np.isfinite(b_cat)
    A_use = A_cat[mask, :]
    b_use = b_cat[mask]

    # 5) 리지 해
    ATA = A_use.T @ A_use
    rhs = A_use.T @ b_use
    ATA[np.diag_indices_from(ATA)] += lambda_ridge
    delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

    # 6) Δcurve 생성
    K = len(self._jac_artifacts["knots"])
    dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
    Phi  = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)
    corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

    high_256_new = {
        "R_High": (high_256["R_High"] + corr_R).astype(np.float32),
        "G_High": (high_256["G_High"] + corr_G).astype(np.float32),
        "B_High": (high_256["B_High"] + corr_B).astype(np.float32),
    }

    # 7) 경계/단조/클램프 → 12bit 업샘플
    for ch in high_256_new:
        self._enforce_monotone(high_256_new[ch])
        high_256_new[ch] = np.clip(high_256_new[ch], 0, 4095)

    lut_new_std = {}
    # Low는 유지
    for ch in ['R_Low','G_Low','B_Low']:
        lut_new_std[ch] = np.array(lut_cur[ch], dtype=np.float32).copy()
    # High만 교체(업샘플)
    for ch in ['R_High','G_High','B_High']:
        lut_new_std[ch] = self._up256_to_4096(high_256_new[ch])

    # 반환: 표준 키 4096pt
    return lut_new_std