# ===== [NEW] TV JSON → 표시/계산용 표준 LUT 키 변환 =====
def _tvjson_to_std_lut(self, vac_dict: dict) -> dict:
    """
    TV JSON의 LUT 키(RchannelLow/High 등)를 표준 키(R_Low/R_High/...)로 변환하여 반환.
    """
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
				
# ===== [NEW] 제어 필드만 추출 =====
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

# ===== [REPLACE] build_vacparam_std_format: cache의 제어필드 + 새 LUT로 JSON 생성 =====
def build_vacparam_std_format(self, lut_new_std: dict) -> str:
    """
    입력: lut_new_std = {R_Low, R_High, G_Low, G_High, B_Low, B_High} 각 4096(또는 근사)
    제어필드: self._vac_cache_std(최근 TV JSON)에서 가져와 합성
    출력: 장치에 바로 쓸 수 있는 JSON 문자열
    """
    if not hasattr(self, "_vac_cache_std") or not isinstance(self._vac_cache_std, dict):
        raise RuntimeError("[VAC build] no cached TV JSON (self._vac_cache_std).")

    # 1) 제어필드 가져오기
    controls = self._extract_control_fields(self._vac_cache_std)

    # 2) LUT 표준키를 TV JSON 키로 만들기 + 4096 보장
    keymap = {
        "R_Low":  "RchannelLow",
        "R_High": "RchannelHigh",
        "G_Low":  "GchannelLow",
        "G_High": "GchannelHigh",
        "B_Low":  "BchannelLow",
        "B_High": "BchannelHigh",
    }
    lut_tv = {}
    for std_k, tv_k in keymap.items():
        if std_k not in lut_new_std:
            raise KeyError(f"[VAC build] missing LUT key: {std_k}")
        lut_tv[tv_k] = self._coerce_lut_4096(lut_new_std[std_k])

    vacparam_out = {**controls, **lut_tv}
    return json.dumps(vacparam_out, ensure_ascii=False, separators=(',', ':'))
				
# ===== [NEW] 보정 사이클: 계산 → JSON 만들기 → 쓰기 → 읽기 → 재측정 =====
def _run_one_correction_cycle(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
    lut_new_std = self._run_correction_iteration(iter_idx=iter_idx,
                                                 max_iters=max_iters,
                                                 lambda_ridge=lambda_ridge)
    if lut_new_std is None:
        logging.error("[CORR] 보정 LUT 계산 실패")
        return

    vac_json = self.build_vacparam_std_format(lut_new_std)  # 제어필드는 self._vac_cache_std에서 자동 사용

    def _after_write(ok, msg):
        logging.info(f"[CORR WRITE] {msg}")
        if not ok:
            return

        def _after_read(vac_dict):
            # 최신 TV JSON 캐시 갱신
            if vac_dict:
                self._vac_cache_std = vac_dict
                try:
                    std_lut = self._tvjson_to_std_lut(vac_dict)
                    self._update_lut_chart_and_table(std_lut)
                except Exception as e:
                    logging.exception(e)

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
