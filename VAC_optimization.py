    def _load_jacobian_artifacts(self):
        """
        jacobian_Y0_high.pkl 파일을 불러와서 artifacts 딕셔너리로 반환
        """
        jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_Y0_high.pkl')
        if not os.path.exists(jac_path):
            logging.error(f"[Jacobian] 파일을 찾을 수 없습니다: {jac_path}")
            raise FileNotFoundError(f"Jacobian model not found: {jac_path}")

        artifacts = joblib.load(jac_path)
        logging.info(f"[Jacobian] 모델 로드 완료: {jac_path}")
        print("======================= artifacts 구조 확인 =======================")
        logging.debug(f"Artifacts keys: {artifacts.keys()}")
        logging.debug(f"Components: {artifacts['components'].keys()}")
        return artifacts
    
    def _build_A_from_artifacts(self, artifacts, comp: str):
        """
        저장된 자코비안 pkl로부터 A 행렬 (ΔY ≈ A·ΔH) 복원
        """
        def stack_basis_all_grays(knots: np.ndarray, L=256) -> np.ndarray:
            """
            모든 그레이(0..255)에 대한 φ(g) K차원 가중치 행렬 (L x K)
            """
            def linear_interp_weights(g: int, knots: np.ndarray) -> np.ndarray:
                """
                그레이 g(0..255)에 대해, K개 knot에 대한 선형보간 '모자(hat)' 가중치 벡터 φ(g) 반환.
                - 양 끝은 1개, 중간은 2개 노드만 비영(희소)
                """
                K = len(knots)
                w = np.zeros(K, dtype=np.float32)
                # 왼쪽/오른쪽 경계
                if g <= knots[0]:
                    w[0] = 1.0
                    return w
                if g >= knots[-1]:
                    w[-1] = 1.0
                    return w
                # 내부: 인접한 두 knot 사이
                i = np.searchsorted(knots, g) - 1
                g0, g1 = knots[i], knots[i+1]
                t = (g - g0) / max(1, (g1 - g0))
                w[i]   = 1.0 - t
                w[i+1] = t
                return w
            
            rows = [linear_interp_weights(g, knots) for g in range(L)]
            return np.vstack(rows).astype(np.float32)

        knots = np.asarray(artifacts["knots"], dtype=np.int32)
        comp_obj = artifacts["components"][comp]
        coef = np.asarray(comp_obj["coef"], dtype=np.float32)

        s = comp_obj["feature_slices"]
        s_high_R = slice(s["high_R"][0], s["high_R"][1])
        s_high_G = slice(s["high_G"][0], s["high_G"][1])
        s_high_B = slice(s["high_B"][0], s["high_B"][1])

        beta_R = coef[s_high_R]
        beta_G = coef[s_high_G]
        beta_B = coef[s_high_B]

        Phi = stack_basis_all_grays(knots, L=256)

        A_R = Phi * beta_R.reshape(1, -1)
        A_G = Phi * beta_G.reshape(1, -1)
        A_B = Phi * beta_B.reshape(1, -1)

        A = np.hstack([A_R, A_G, A_B]).astype(np.float32)
        logging.info(f"[Jacobian] {comp} A 행렬 shape: {A.shape}") # (256, 3K)
        return A
    
    def _run_off_baseline_then_on(self):
        # 감마 라인 준비(“VAC OFF (Ref.)”)
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
            # (3) DB에서 모델/주사율에 맞는 VAC Data 적용 → 읽기 → LUT 차트 갱신
            self._apply_vac_from_db_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off, gamma_lines=gamma_lines_off,
            gray_levels=list(range(256)), patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_off
        )
        
    def _apply_vac_from_db_and_measure_on(self):
        # (A) DB에서 panel/frame_rate로 VAC_Data 가져오기
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error("매칭되는 VAC Data가 없습니다.")
            return

        # (B) TV에 적용 → 읽기 (기존 Thread 재사용 또는 직접 RPC)
        ok = self._write_vac_to_tv(vac_data)
        if not ok:
            logging.error("VAC Data 쓰기 실패")
        read_back = self._read_vac_from_tv()
        if read_back:
            self._update_lut_chart_and_table(read_back)  # self.vac_optimization_lut_chart, self.ui.vac_table_rbgLUT_4

        # (C) VAC ON 측정
        gamma_lines_on = {
            'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label="VAC ON") for p in ('white','red','green','blue')},
            'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label="VAC ON") for p in ('white','red','green','blue')},
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
                logging.info("스펙 통과 — 종료")
                return
            # (D) 보정 반복 시작
            self._run_correction_iteration(iter_idx=1)

        self.start_viewing_angle_session(
            profile=profile_on, gamma_lines=gamma_lines_on,
            gray_levels=list(range(256)), patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_on
        )
        
    def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
        logging.info(f"[CORR] iteration {iter_idx} start")

        # 1) 현재 TV LUT 읽기
        lut_cur = self._read_vac_from_tv()
        if lut_cur is None:
            logging.error("현재 LUT 읽기 실패")
            return

        # 2) 4096→256 다운샘플 (High만 수정, Low 고정)
        high_256 = {ch: self._down4096_to_256(lut_cur[ch]) for ch in ['R_High','G_High','B_High']}
        low_256  = {ch: self._down4096_to_256(lut_cur[ch]) for ch in ['R_Low','G_Low','B_Low']}

        # 3) Δ 목표(white/main 기준): OFF vs ON 차이를 256 길이로 구성
        #    Gamma: 1..254 유효, Cx/Cy: 0..255
        d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
        # d_targets: {"Gamma":(256,), "Cx":(256,), "Cy":(256,)}

        # 4) 자코비안 A (Gamma/Cx/Cy 각각 256 x (3K))
        A_G = self._build_A_from_artifacts(self._jac_artifacts, "Gamma")  # (256, 3K)
        A_Cx = self._build_A_from_artifacts(self._jac_artifacts, "Cx")
        A_Cy = self._build_A_from_artifacts(self._jac_artifacts, "Cy")

        # 5) 결합 선형계: [wG*A_G; wC*A_Cx; wC*A_Cy] Δh = - [wG*ΔGamma; wC*ΔCx; wC*ΔCy]
        wG, wC = 1.0, 1.0
        A_cat = np.vstack([wG*A_G, wC*A_Cx, wC*A_Cy]).astype(np.float32)
        b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)

        # 유효치 마스크(특히 gamma의 NaN)
        mask = np.isfinite(b_cat)
        A_use = A_cat[mask, :]
        b_use = b_cat[mask]

        # 6) 리지 해(Δh) 구하기 (3K-dim: [Rknots, Gknots, Bknots])
        #    (A^T A + λI) Δh = A^T b
        ATA = A_use.T @ A_use
        rhs = A_use.T @ b_use
        ATA[np.diag_indices_from(ATA)] += lambda_ridge
        delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

        # 7) Δcurve = Phi * Δh_channel 로 256-포인트 보정곡선 만들고 High에 적용
        K = len(self._jac_artifacts["knots"])
        dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
        Phi  = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)
        corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

        high_256_new = {
            "R_High": (high_256["R_High"] + corr_R).astype(np.float32),
            "G_High": (high_256["G_High"] + corr_G).astype(np.float32),
            "B_High": (high_256["B_High"] + corr_B).astype(np.float32),
        }

        # 8) 경계/단조/클램프 → 12bit 업샘플
        for ch in high_256_new:
            self._enforce_monotone(high_256_new[ch])
            high_256_new[ch] = np.clip(high_256_new[ch], 0, 4095)

        lut_new = {}
        # Low는 고정
        for ch in ['R_Low','G_Low','B_Low']:
            lut_new[ch] = lut_cur[ch].copy()
        # High는 보정 결과로 교체(4096 업샘플)
        for ch in ['R_High','G_High','B_High']:
            lut_new[ch] = self._up256_to_4096(high_256_new[ch])

        # 9) TV에 적용 → 읽기 → LUT 차트 갱신
        ok = self._write_vac_to_tv(lut_new)
        if not ok:
            logging.error("보정 LUT 쓰기 실패")
        read_back = self._read_vac_from_tv()
        if read_back:
            self._update_lut_chart_and_table(read_back)

        # 10) 재측정 (CORR i)
        label = f"CORR{iter_idx}"
        gamma_lines_corr = {
            'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=label) for p in ('white','red','green','blue')},
            'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=label) for p in ('white','red','green','blue')},
        }
        profile_corr = SessionProfile(
            legend_text=label,
            cie_label=f"data_{iter_idx+2}",
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

    def _after_corr(store_corr):
        self._on_store = store_corr  # 최신을 갱신
        if self._check_spec_pass(self._off_store, self._on_store):
            logging.info("스펙 통과 — 종료")
            return
        if iter_idx >= max_iters:
            logging.info("최대 보정 횟수 도달 — 종료")
            return
        self._run_correction_iteration(iter_idx+1, max_iters=max_iters)

        self.start_viewing_angle_session(
            profile=profile_corr, gamma_lines=gamma_lines_corr,
            gray_levels=list(range(256)), patterns=('white','red','green','blue'),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_corr
        )
    
    def _check_spec_pass(self, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
    # white/main만 기준
        def _extract_white(series_store):
            lv = np.zeros(256); cx = np.zeros(256); cy = np.zeros(256)
            for g in range(256):
                tup = series_store['gamma']['main']['white'].get(g, None)
                if tup: lv[g], cx[g], cy[g] = tup
                else:   lv[g]=np.nan; cx[g]=np.nan; cy[g]=np.nan
            return lv, cx, cy

        lv_ref, cx_ref, cy_ref = _extract_white(off_store)
        lv_on , cx_on , cy_on  = _extract_white(on_store)

        G_ref = self._compute_gamma_series(lv_ref)
        G_on  = self._compute_gamma_series(lv_on)

        dG  = np.abs(G_on - G_ref)
        dCx = np.abs(cx_on - cx_ref)
        dCy = np.abs(cy_on - cy_ref)

        max_dG  = np.nanmax(dG)
        max_dCx = np.nanmax(dCx)
        max_dCy = np.nanmax(dCy)

        logging.info(f"[SPEC] max|ΔGamma|={max_dG:.6f} (≤{thr_gamma}), max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c})")
        return (max_dG <= thr_gamma) and (max_dCx <= thr_c) and (max_dCy <= thr_c)

    def _build_delta_targets_from_stores(self, off_store, on_store):
        # Δ = (ON - OFF). white/main
        lv_ref, cx_ref, cy_ref = np.zeros(256), np.zeros(256), np.zeros(256)
        lv_on , cx_on , cy_on  = np.zeros(256), np.zeros(256), np.zeros(256)
        for g in range(256):
            tR = off_store['gamma']['main']['white'].get(g, None)
            tO = on_store['gamma']['main']['white'].get(g, None)
            if tR: lv_ref[g], cx_ref[g], cy_ref[g] = tR
            else:  lv_ref[g]=np.nan; cx_ref[g]=np.nan; cy_ref[g]=np.nan
            if tO: lv_on[g], cx_on[g], cy_on[g] = tO
            else:  lv_on[g]=np.nan; cx_on[g]=np.nan; cy_on[g]=np.nan

        G_ref = self._compute_gamma_series(lv_ref)
        G_on  = self._compute_gamma_series(lv_on)
        d = {
            "Gamma": (G_on - G_ref),
            "Cx":    (cx_on - cx_ref),
            "Cy":    (cy_on - cy_ref),
        }
        # NaN → 0 (선형계 마스킹에서도 걸러지니 안정성↑)
        for k in d:
            d[k] = np.nan_to_num(d[k], nan=0.0).astype(np.float32)
        return d
    
    def start_viewing_angle_session(
        self,
        profile: SessionProfile,
        gamma_lines: dict,
        gray_levels=None,
        patterns=('white','red','green','blue'),
        colorshift_patterns=None,
        first_gray_delay_ms=3000,
        cs_settle_ms=1000,
        on_done=None,
    ):
        if gray_levels is None:
            # gray_levels = list(range(256))
            gray_levels = op.gray_levels
        if colorshift_patterns is None:
            colorshift_patterns = op.colorshift_patterns  # 기존 상수 사용

        store = {
            'gamma': {'main': {p:{} for p in patterns}, 'sub': {p:{} for p in patterns}},
            'colorshift': {'main': [], 'sub': []}
        }

        self._sess = {
            'phase': 'gamma',
            'p_idx': 0,
            'g_idx': 0,
            'cs_idx': 0,
            'patterns': list(patterns),
            'gray_levels': list(gray_levels),
            'cs_patterns': colorshift_patterns,
            'store': store,
            'profile': profile,
            'gamma_lines': gamma_lines,
            'first_gray_delay_ms': first_gray_delay_ms,
            'cs_settle_ms': cs_settle_ms,
            'on_done': on_done
        }
        self._session_step()

    def _session_step(self):
        s = self._sess
        if s['phase'] == 'gamma':
            if s['p_idx'] >= len(s['patterns']):
                s['phase'] = 'colorshift'
                s['cs_idx'] = 0
                QTimer.singleShot(60, lambda: self._session_step())
                return

            if s['g_idx'] >= len(s['gray_levels']):
                s['g_idx'] = 0
                s['p_idx'] += 1
                QTimer.singleShot(40, lambda: self._session_step())
                return

            pattern = s['patterns'][s['p_idx']]
            gray = s['gray_levels'][s['g_idx']]

            if pattern == 'white':
                rgb_value = f"{gray},{gray},{gray}"
            elif pattern == 'red':
                rgb_value = f"{gray},0,0"
            elif pattern == 'green':
                rgb_value = f"0,{gray},0"
            else:
                rgb_value = f"0,0,{gray}"
            self.changeColor(rgb_value)

            delay = s['first_gray_delay_ms'] if s['g_idx'] == 0 else 0
            QTimer.singleShot(delay, lambda p=pattern, g=gray: self._trigger_gamma_pair(p, g))

        elif s['phase'] == 'colorshift':
            if s['cs_idx'] >= len(s['cs_patterns']):
                s['phase'] = 'done'
                QTimer.singleShot(0, lambda: self._session_step())
                return

            pname, r, g, b = s['cs_patterns'][s['cs_idx']]
            self.changeColor(f"{r},{g},{b}")
            QTimer.singleShot(s['cs_settle_ms'], lambda pn=pname: self._trigger_colorshift_pair(pn))

        else:  # done
            self._finalize_session()

    def _trigger_gamma_pair(self, pattern, gray):
        s = self._sess
        s['_gamma'] = {}

        def handle(role, res):
            s['_gamma'][role] = res
            got_main = 'main' in s['_gamma']
            got_sub = ('sub') in s ['_gamma'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_gamma_pair(pattern, gray, s['_gamma'])
                s['g_idx'] += 1
                QTimer.singleShot(30, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle)
            self.sub_measure_thread.start()

    def _consume_gamma_pair(self, pattern, gray, results):
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']
        lines = s['gamma_lines']

        for role in ('main', 'sub'):
            if role not in results or results[role] is None:
                store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
                continue
            x, y, lv, cct, duv = results[role]
            store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))
            # Lv vs gray
            line = lines[role][pattern]
            xs = list(line.get_xdata()); ys = list(line.get_ydata())
            xs.append(gray); ys.append(float(lv))
            line.set_data(xs, ys)

        # white/main만 테이블 기록
        if pattern == 'white':
            lv_m, cx_m, cy_m = store['gamma']['main'][pattern][gray]
            table = self.ui.vac_table_measure_results_main_2
            cols = profile.table_cols
            self._set_item(table, gray, cols['lv'], f"{lv_m:.6f}")
            self._set_item(table, gray, cols['cx'], f"{cx_m:.6f}")
            self._set_item(table, gray, cols['cy'], f"{cy_m:.6f}")

            if profile.ref_store is not None:
                ref = profile.ref_store['gamma']['main']['white'].get(gray, None)
                if ref is not None and 'd_cx' in cols and 'd_cy' in cols:
                    lv_r, cx_r, cy_r = ref
                    self._set_item(table, gray, cols['d_cx'], f"{(cx_m - cx_r):.6f}")
                    self._set_item(table, gray, cols['d_cy'], f"{(cy_m - cy_r):.6f}")

        self.vac_optimization_gamma_chart.autoscale()
        self.vac_optimization_gamma_chart.draw()

    def _trigger_colorshift_pair(self, patch_name):
        s = self._sess
        s['_cs'] = {}

        def handle(role, res):
            s['_cs'][role] = res
            got_main = 'main' in s['_cs']
            got_sub = ('sub') in s ['_cs'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_colorshift_pair(patch_name, s['_cs'])
                s['cs_idx'] += 1
                QTimer.singleShot(80, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(lambda r: handle('main', r))
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(lambda r: handle('sub', r))
            self.sub_measure_thread.start()

    def _consume_colorshift_pair(self, patch_name, results):
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        for role in ('main', 'sub'):
            if role not in results or results[role] is None:
                store['colorshift'][role].append((np.nan, np.nan, np.nan, np.nan))
                continue
            x, y, lv, cct, duv = results[role]
            u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))
            store['colorshift'][role].append((float(x), float(y), float(u_p), float(v_p)))
            angle = 0 if role == 'main' else 60
            self.vac_optimization_colorshift_chart.update(
                u_p, v_p, data_label=profile.cie_label, view_angle=angle, vac_status=profile.legend_text
            )

    def _finalize_session(self):
        s = self._sess
        profile: SessionProfile = s['profile']
        table = self.ui.vac_table_measure_results_main_2
        cols = profile.table_cols

        # white/main 감마 계산
        lv_series = np.zeros(256, dtype=np.float64)
        for g in range(256):
            tup = s['store']['gamma']['main']['white'].get(g, None)
            lv_series[g] = float(tup[0]) if tup else np.nan
        gamma_vec = self._compute_gamma_series(lv_series)
        for g in range(256):
            if np.isfinite(gamma_vec[g]):
                self._set_item(table, g, cols['gamma'], f"{gamma_vec[g]:.6f}")

        # ΔGamma (ON/보정 시)
        if profile.ref_store is not None and 'd_gamma' in cols:
            ref_lv = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = profile.ref_store['gamma']['main']['white'].get(g, None)
                ref_lv[g] = float(tup[0]) if tup else np.nan
            ref_gamma = self._compute_gamma_series(ref_lv)
            dG = gamma_vec - ref_gamma
            for g in range(256):
                if np.isfinite(dG[g]):
                    self._set_item(table, g, cols['d_gamma'], f"{dG[g]:.6f}")

        if callable(s['on_done']):
            try:
                s['on_done'](s['store'])
            except Exception as e:
                logging.exception(e)
                
    def _ensure_row_count(self, table, row_idx):
        if table.rowCount() <= row_idx:
            table.setRowCount(row_idx + 1)

    def _set_item(self, table, row, col, value):
        self._ensure_row_count(table, row)
        table.setItem(row, col, QTableWidgetItem("" if value is None else str(value)))

    def _compute_gamma_series(self, lv_vec_256):
        lv = np.asarray(lv_vec_256, dtype=np.float64)
        gamma = np.full(256, np.nan, dtype=np.float64)
        lv0 = lv[0]
        denom = np.max(lv[1:] - lv0)
        if not np.isfinite(denom) or denom <= 0:
            return gamma
        nor = (lv - lv0) / denom
        gray = np.arange(256, dtype=np.float64)
        gray_norm = gray / 255.0
        valid = (gray >= 1) & (gray <= 254) & (nor > 0) & np.isfinite(nor)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma[valid] = np.log(nor[valid]) / np.log(gray_norm[valid])
        return gamma
    
    def _stack_basis(self, knots, L=256):
        knots = np.asarray(knots, dtype=np.int32)
        
        def _phi(g):
            # 선형 모자 함수
            K = len(knots)
            w = np.zeros(K, dtype=np.float32)
            if g <= knots[0]:
                w[0]=1.; return w
            if g >= knots[-1]:
                w[-1]=1.; return w
            i = np.searchsorted(knots, g) - 1
            g0, g1 = knots[i], knots[i+1]
            t = (g - g0) / max(1, (g1 - g0))
            w[i] = 1-t; w[i+1] = t
            return w
        return np.vstack([_phi(g) for g in range(L)])

    def _down4096_to_256(self, arr4096):
        arr4096 = np.asarray(arr4096, dtype=np.float32)
        idx = np.round(np.linspace(0, 4095, 256)).astype(int)
        return arr4096[idx]

    def _up256_to_4096(self, arr256):
        arr256 = np.asarray(arr256, dtype=np.float32)
        x_small = np.linspace(0, 1, 256)
        x_big   = np.linspace(0, 1, 4096)
        return np.interp(x_big, x_small, arr256).astype(np.float32)

    def _enforce_monotone(self, arr):
        # 제자리 누적 최대치
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                arr[i] = arr[i-1]
        return arr
        
    def _fetch_vac_by_model(self, panel_maker, frame_rate):
        """
        DB: W_VAC_Application_Status에서 Panel_Maker/Frame_Rate 매칭 → VAC_Info_PK 얻고
            W_VAC_Info.PK=VAC_Info_PK → VAC_Data 읽어서 반환
        반환: (pk, vac_version, vac_data_dict)  또는 (None,None,None)
        """
        try:
            # TODO: 실제 DB 조회 코드로 교체
            pk = ...; vac_version = ...; vac_data = ...  # dict with keys: R_Low,R_High,G_Low, ...
            return pk, vac_version, vac_data
        except Exception as e:
            logging.exception(e)
            return None, None, None

    def _write_vac_to_tv(self, vac_data):
        """
        이미 쓰시던 WriteVACdataThread/명령 기반으로 연결.
        vac_data(dict): { 'R_Low':[4096], 'R_High':[4096], ... }
        """
        try:
            # TODO: 실제 쓰기
            return True
        except Exception as e:
            logging.exception(e)
            return False

    def _read_vac_from_tv(self):
        """
        기존 ReadVACdataThread 결과와 동일 구조로 반환
        return dict or None
        """
        try:
            # TODO: 실제 읽기
            # 예시 keys: {'R_Low':[4096], 'R_High':[4096], 'G_Low':...}
            return ...
        except Exception as e:
            logging.exception(e)
            return None

    def _update_lut_chart_and_table(self, lut_dict):
        """
        self.vac_optimization_lut_chart (x:0~4095) 갱신 + self.ui.vac_table_rbgLUT_4에 숫자 뿌리기
        이미 사용중인 update_rgbchannel_chart/update_rgbchannel_table 재사용해도 됩니다.
        """
        try:
            import pandas as pd
            df = pd.DataFrame({
                "R_Low":  lut_dict["R_Low"],  "R_High": lut_dict["R_High"],
                "G_Low":  lut_dict["G_Low"],  "G_High": lut_dict["G_High"],
                "B_Low":  lut_dict["B_Low"],  "B_High": lut_dict["B_High"],
            })
            # 예: 기존 메서드 재사용
            self.update_rgbchannel_chart(
                df,
                self.graph['vac_laboratory']['data_acquisition_system']['input']['ax'],
                self.graph['vac_laboratory']['data_acquisition_system']['input']['canvas']
            )
            self.update_rgbchannel_table(df, self.ui.vac_table_rbgLUT_4)
        except Exception as e:
            logging.exception(e)

    def start_VAC_optimization(self):
        """
        =============== 메인 엔트리: 버튼 이벤트 연결용 ===============
        전체 플로우:
        1) VAC OFF 보장 → 측정(OFF baseline) + UI 업데이트
        2) DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 적용(ON) → 측정(ON 현재) + UI 업데이트
        3) 스펙 확인 → 통과면 종료
        4) 미통과면 자코비안 기반 보정(256기준) → 4096 보간 반영 → TV 적용 → 재측정 → 스펙 재확인
        5) (필요 시 반복 2~3회만)
        """
        try:
            # (0) 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            print("======================= A 행렬 shape 확인 =======================")
            self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")
            self.A_Cx    = self._build_A_from_artifacts(artifacts, "Cx")
            self.A_Cy    = self._build_A_from_artifacts(artifacts, "Cy")

        except FileNotFoundError as e:
            logging.error(f"[VAC Optimization] Jacobian file not found: {e}")
            print("❌ 자코비안 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")

        except KeyError as e:
            logging.error(f"[VAC Optimization] Missing key in artifacts: {e}")
            print(f"❌ artifacts 딕셔너리에 '{e}' 키가 없습니다. 자코비안 파일 구조를 확인해주세요.")

        except Exception as e:
            logging.exception("[VAC Optimization] Unexpected error occurred")
            print(f"❌ 예기치 못한 오류가 발생했습니다: {e}")

        # (1) VAC OFF 보장 + 측정
        # 결과 저장용 버퍼 초기화 (OFF / ON 구분)
        self._off_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        self._on_store  = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        # TV VAC OFF 하기
        st = self.check_VAC_status()
        if st.get("activated", False):
            logging.debug("VAC 활성 상태 → OFF로 전환 시도")
            self.send_command(self.ser_tv, 's')
            cmd = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff":false}\''
            self.send_command(self.ser_tv, cmd)
            self.send_command(self.ser_tv, 'exit')
            st2 = self.check_VAC_status()
            if st2.get("activated", False):
                logging.warning("VAC OFF 실패로 보입니다. 그래도 측정 진행")
            else:
                logging.info("VAC OFF 전환 성공.")
        else:
            logging.debug("이미 VAC OFF 상태. OFF 레퍼런스 측정 진행")
            
        # (2) OFF 측정 세션 시작
        self._run_off_baseline_then_on()
