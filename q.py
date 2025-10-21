def _consume_gamma_pair(self, pattern, gray, results):
    """
    results: {
      'main': (x, y, lv, cct, duv)  또는  None,
      'sub' : (x, y, lv, cct, duv)  또는  None
    }
    """
    s = self._sess
    store = s['store']
    profile: SessionProfile = s['profile']

    # 현재 세션이 OFF 레퍼런스인지, ON/보정 런인지 상태 문자열 결정
    state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

    # 두 역할을 results 키로 직접 순회 (측정기 객체 비교 X)
    for role in ('main', 'sub'):
        res = results.get(role, None)
        if res is None:
            # 측정 실패/결측인 경우
            store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
            continue

        x, y, lv, cct, duv = res
        # 스토어 업데이트 (white 테이블/감마 계산 등에 사용)
        store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

        # ▶▶ 차트 업데이트 (간소화 API)
        # GammaChartVAC: add_point(state, role, pattern, gray, luminance)
        self.vac_optimization_gamma_chart.add_point(
            state=state,
            role=role,               # 'main' 또는 'sub'
            pattern=pattern,         # 'white'|'red'|'green'|'blue'
            gray=int(gray),
            luminance=float(lv)
        )

    # (아래 white/main 테이블 채우는 로직은 기존 그대로 유지)
    if pattern == 'white':
        # main 테이블
        lv_m, cx_m, cy_m = store['gamma']['main']['white'].get(gray, (np.nan, np.nan, np.nan))
        table_inst1 = self.ui.vac_table_opt_mes_results_main
        cols = profile.table_cols
        self._set_item(table_inst1, gray, cols['lv'], f"{lv_m:.6f}" if np.isfinite(lv_m) else "")
        self._set_item(table_inst1, gray, cols['cx'], f"{cx_m:.6f}" if np.isfinite(cx_m) else "")
        self._set_item(table_inst1, gray, cols['cy'], f"{cy_m:.6f}" if np.isfinite(cy_m) else "")

        # sub 테이블
        lv_s, cx_s, cy_s = store['gamma']['sub']['white'].get(gray, (np.nan, np.nan, np.nan))
        table_inst2 = self.ui.vac_table_opt_mes_results_sub
        self._set_item(table_inst2, gray, cols['lv'], f"{lv_s:.6f}" if np.isfinite(lv_s) else "")
        self._set_item(table_inst2, gray, cols['cx'], f"{cx_s:.6f}" if np.isfinite(cx_s) else "")
        self._set_item(table_inst2, gray, cols['cy'], f"{cy_s:.6f}" if np.isfinite(cy_s) else "")

        # ΔCx/ΔCy (ON 세션에서만; ref_store가 있을 때)
        if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
            ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
            if ref_main is not None:
                _, cx_r, cy_r = ref_main
                if np.isfinite(cx_m): self._set_item(table_inst1, gray, cols['d_cx'], f"{(cx_m - cx_r):.6f}")
                if np.isfinite(cy_m): self._set_item(table_inst1, gray, cols['d_cy'], f"{(cy_m - cy_r):.6f}")

            ref_sub = profile.ref_store['gamma']['sub']['white'].get(gray, None)
            if ref_sub is not None:
                _, cx_r_s, cy_r_s = ref_sub
                if np.isfinite(cx_s): self._set_item(table_inst2, gray, cols['d_cx'], f"{(cx_s - cx_r_s):.6f}")
                if np.isfinite(cy_s): self._set_item(table_inst2, gray, cols['d_cy'], f"{(cy_s - cy_r_s):.6f}")
                    
def _consume_colorshift_pair(self, patch_name, results):
    """
    results: {
      'main': (x, y, lv, cct, duv)  또는  None,
      'sub' : (x, y, lv, cct, duv)  또는  None
    }
    """
    s = self._sess
    store = s['store']
    profile: SessionProfile = s['profile']

    # 현재 세션 상태
    state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

    for role in ('main', 'sub'):
        res = results.get(role, None)
        if res is None:
            store['colorshift'][role].append((np.nan, np.nan, np.nan, np.nan))
            continue

        x, y, lv, cct, duv = res

        # xy → u′v′ 변환
        u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))
        store['colorshift'][role].append((float(x), float(y), float(u_p), float(v_p)))

        # ▶▶ 차트 업데이트 (간소화 API)
        # CIE1976ChartVAC: add_point(state, role, u_p, v_p)
        self.vac_optimization_cie1976_chart.add_point(
            state=state,
            role=role,        # 'main' 또는 'sub'
            u_p=float(u_p),
            v_p=float(v_p)
        )
        
def _apply_vac_from_db_and_measure_on(self):
    """
    3-a) DB에서 Panel_Maker + Frame_Rate 조합인 VAC_Data 가져오기
    3-b) TV에 쓰기 → TV에서 읽기
        → LUT 차트 갱신(reset_and_plot)
        → ON 시리즈 리셋(reset_on)
        → ON 측정 세션 시작(start_viewing_angle_session)
    """
    # 3-a) DB에서 VAC JSON 로드
    panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
    fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
    vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
    if vac_data is None:
        logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 종료")
        return

    # TV 쓰기 완료 시 콜백
    def _after_write(ok, msg):
        logging.info(f"[VAC Write] {msg}")
        if not ok:
            logging.error("VAC Writing 실패 - 종료")
            return
        # 쓰기 성공 → TV에서 VAC 읽어오기
        self._read_vac_from_tv(_after_read)

    # TV에서 읽기 완료 시 콜백
    def _after_read(vac_dict):
        if not vac_dict:
            logging.error("VAC 데이터 읽기 실패 - 종료")
            return

        # 캐시 보관 (TV 원 키명 유지)
        self._vac_dict_cache = vac_dict

        # LUT 차트는 "받을 때마다 전체 리셋 후 재그림"
        # TV 키명을 표준 표시용으로 바꿔서 전달 (RchannelHigh -> R_High 등)
        lut_plot = {
            key.replace("channel", "_"): v
            for key, v in vac_dict.items()
            if "channel" in key
        }
        # 새 LUT로 전체 리셋 후 플로팅
        self.vac_optimization_lut_chart.reset_and_plot(lut_plot)

        # ── ON 세션 시작 전: ON 시리즈 전부 리셋 ──
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        # ON 세션 프로파일 (OFF를 참조로 Δ 계산)
        profile_on = SessionProfile(
            legend_text="VAC ON",
            cie_label="data_2",
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

        # ON 세션 종료 후: 스펙 체크 → 미통과면 보정 1회차 진입
        def _after_on(store_on):
            self._on_store = store_on
            if self._check_spec_pass(self._off_store, self._on_store):
                logging.info("✅ 스펙 통과 — 종료")
                return
            # (D) 반복 보정 시작 (1회차)
            self._run_correction_iteration(iter_idx=1)

        # ── ON 측정 세션 시작 ──
        # 간소화된 API: gamma_lines 인자 제거
        self.start_viewing_angle_session(
            profile=profile_on,
            gray_levels=getattr(op, "gray_levels_256", list(range(256))),
            gamma_patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            cs_settle_ms=1000,
            on_done=_after_on
        )

    # 3-b) VAC_Data TV에 writing
    self._write_vac_to_tv(vac_data, on_finished=_after_write)