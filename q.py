    def _apply_vac_from_db_and_measure_on(self):
        """
        3-a) DB에서 Panel_Maker + Frame_Rate 조합인 VAC_Data 가져오기
        3-b) TV에 쓰기 → TV에서 읽기
            → LUT 차트 갱신(reset_and_plot)
            → ON 시리즈 리셋(reset_on)
            → ON 측정 세션 시작(start_viewing_angle_session)
        """
        # 3-a) DB에서 VAC JSON 로드
        self._step_start(2)
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
            return
        self._step_done(2)
        
        # # [ADD] 런타임 X(256×18) 생성 & 스키마 디버그 로깅
        # try:
        #     X_runtime, lut256_norm, ctx = self._build_runtime_X_from_db_json(vac_data)
        #     self._debug_log_runtime_X(X_runtime, ctx, tag="[RUNTIME X from DB+UI]")
        # except Exception as e:
        #     logging.exception("[RUNTIME X] build/debug failed")
        #     # 여기서 실패하면 예측/최적화 전에 스키마 문제로 조기 중단하도록 권장
        #     return
        
        # # ✅ 0) OFF 끝났고, 여기서 1차 예측 최적화 먼저 수행
        # logging.info("[PredictOpt] 예측 기반 1차 최적화 시작")
        # vac_data_by_predict, _lut4096_dict = self._predictive_first_optimize(
        #     vac_data, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3
        # )
        # if vac_data_by_predict is None:
        #     logging.warning("[PredictOpt] 실패 → 원본 DB LUT로 진행")
        #     vac_data_by_predict = vac_data
        # else:
        #     logging.info("[PredictOpt] 1차 최적화 LUT 생성 완료 → UI 업데이트 반영됨")

        # TV 쓰기 완료 시 콜백
        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[LUT LOADING] DB fetch LUT TV Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            # 쓰기 성공 → TV에서 VAC 읽어오기
            logging.info(f"[LUT LOADING] DB fetch LUT TV Writing 완료: {msg}")
            logging.info("[LUT LOADING] DB fetch LUT TV Writing 확인을 위한 TV Reading 시작")
            self._read_vac_from_tv(_after_read)

        # TV에서 읽기 완료 시 콜백
        def _after_read(vac_dict):
            if not vac_dict:
                logging.error("[LUT LOADING] DB fetch LUT TV Writing 확인을 위한 TV Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[LUT LOADING] DB fetch LUT TV Writing 확인을 위한 TV Reading 완료")
            self._step_done(3)
            # 캐시 보관 (TV 원 키명 유지)
            self._vac_dict_cache = vac_dict
            lut_dict_plot = {key.replace("channel", "_"): v
                            for key, v in vac_dict.items() if "channel" in key
            }
            self._update_lut_chart_and_table(lut_dict_plot)

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
                self._step_done(4)
                self._on_store = store_on
                logging.debug(f"VAC ON 측정 결과:\n{self._on_store}")
                
                self._step_start(5)
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=1, max_iters=2))
                self._spec_thread.start()

            # ── ON 측정 세션 시작 ──
            self._step_start(4)
            logging.info("[MES] DB fetch LUT 기준 측정 시작")
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000, cs_settle_ms=1000,
                on_done=_after_on
            )

        # 3-b) VAC_Data TV에 writing
        logging.info("[LUT LOADING] DB fetch LUT TV Writing 시작")
        self._write_vac_to_tv(vac_data, on_finished=_after_write)

    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        try:
            if metrics and "error" not in metrics:
                logging.info(
                    f"[SPEC(thread)] max|ΔGamma|={metrics['max_dG']:.6f} (≤{metrics['thr_gamma']}), "
                    f"max|ΔCx|={metrics['max_dCx']:.6f}, max|ΔCy|={metrics['max_dCy']:.6f} (≤{metrics['thr_c']})"
                )
            else:
                logging.warning("[SPEC(thread)] evaluation failed — treating as not passed.")

            # 결과 표/차트 갱신
            self._update_spec_views(self._off_store, self._on_store)

            if spec_ok:
                # ✅ 통과: Step5 = complete
                self._step_done(5)
                logging.info("✅ 스펙 통과 — 최적화 종료")
                return

            # ❌ 실패: Step5 = fail
            self._step_fail(5)

            # 다음 보정 루프
            if iter_idx < max_iters:
                logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
                for s in (2,3,4):
                    self._step_set_pending(s)
                self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
            else:
                logging.info("⛔ 최대 보정 횟수 도달 — 종료")
        finally:
            self._spec_thread = None

    def _update_spec_views(self, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
        """
        요구하신 6개 위젯을 모두 갱신:
        1) vac_table_chromaticityDiff  (ΔCx/ΔCy/ΔGamma pass/total)
        2) vac_chart_chromaticityDiff  (Cx,Cy vs gray: OFF/ON)
        3) vac_table_gammaLinearity    (OFF/ON, 88~232 구간별 슬로프 평균)
        4) vac_chart_gammaLinearity    (8gray 블록 평균 슬로프 dot+line)
        5) vac_table_colorShift_3      (4 skin 패턴 Δu′v′, OFF/ON, 평균)
        6) vac_chart_colorShift_3      (Grouped bars)
        """
        # ===== 공통: white/main 시리즈 추출 =====
        def _extract_white(series_store):
            lv = np.full(256, np.nan, np.float64)
            cx = np.full(256, np.nan, np.float64)
            cy = np.full(256, np.nan, np.float64)
            for g in range(256):
                tup = series_store['gamma']['main']['white'].get(g, None)
                if tup:
                    lv[g], cx[g], cy[g] = float(tup[0]), float(tup[1]), float(tup[2])
            return lv, cx, cy

        lv_off, cx_off, cy_off = _extract_white(off_store)
        lv_on , cx_on , cy_on  = _extract_white(on_store)

        # ===== 1) ChromaticityDiff 표: pass/total =====
        G_off = self._compute_gamma_series(lv_off)
        G_on  = self._compute_gamma_series(lv_on)
        dG  = np.abs(G_on - G_off)        # (256,)
        dCx = np.abs(cx_on - cx_off)
        dCy = np.abs(cy_on - cy_off)

        def _pass_total(arr, thr):
            mask = np.isfinite(arr)
            tot = int(np.sum(mask))
            ok  = int(np.sum((np.abs(arr[mask]) <= thr)))
            return ok, tot

        ok_cx, tot_cx = _pass_total(dCx, thr_c)
        ok_cy, tot_cy = _pass_total(dCy, thr_c)
        ok_g , tot_g  = _pass_total(dG , thr_gamma)

        # 표: (제목/헤더 제외) 2열×(2~4행) 채우기
        def _set_text(tbl, row, col, text):
            self._ensure_row_count(tbl, row)
            item = tbl.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                tbl.setItem(row, col, item)
            item.setText(text)

        tbl_ch = self.ui.vac_table_chromaticityDiff
        _set_text(tbl_ch, 1, 1, f"{ok_cx}/{tot_cx}")   # 2행,2열 ΔCx
        _set_text(tbl_ch, 2, 1, f"{ok_cy}/{tot_cy}")   # 3행,2열 ΔCy
        _set_text(tbl_ch, 3, 1, f"{ok_g}/{tot_g}")     # 4행,2열 ΔGamma

        # ===== 2) ChromaticityDiff 차트: Cx/Cy vs gray (OFF/ON) =====
        x = np.arange(256)
        self.vac_optimization_chromaticity_chart.set_series(
            "OFF_Cx", x, cx_off, marker=None, linestyle='-', label='OFF Cx'
        )
        self.vac_optimization_chromaticity_chart.set_series(
            "ON_Cx",  x, cx_on,  marker=None, linestyle='--', label='ON Cx'
        )
        self.vac_optimization_chromaticity_chart.set_series(
            "OFF_Cy", x, cy_off, marker=None, linestyle='-', label='OFF Cy'
        )
        self.vac_optimization_chromaticity_chart.set_series(
            "ON_Cy",  x, cy_on,  marker=None, linestyle='--', label='ON Cy'
        )

        # ===== 3) GammaLinearity 표: 88~232, 8gray 블록 평균 슬로프 =====
        def _segment_mean_slopes(lv_vec, g_start=88, g_end=232, step=8):
            # 1-step slope 정의(동일): 255*(lv[g+1]-lv[g])
            one_step = 255.0*(lv_vec[1:]-lv_vec[:-1])  # 0..254 개
            means = []
            for g in range(g_start, g_end, step):      # 88,96,...,224
                block = one_step[g:g+step]             # 8개
                m = np.nanmean(block) if np.isfinite(block).any() else np.nan
                means.append(m)
            return np.array(means, dtype=np.float64)   # 길이 = (232-88)/8 = 18개

        m_off = _segment_mean_slopes(lv_off)
        m_on  = _segment_mean_slopes(lv_on)
        avg_off = float(np.nanmean(m_off)) if np.isfinite(m_off).any() else float('nan')
        avg_on  = float(np.nanmean(m_on))  if np.isfinite(m_on).any()  else float('nan')

        tbl_gl = self.ui.vac_table_gammaLinearity
        _set_text(tbl_gl, 1, 1, f"{avg_off:.6f}")  # 2행,2열 OFF 평균 기울기
        _set_text(tbl_gl, 1, 2, f"{avg_on:.6f}")   # 2행,3열 ON  평균 기울기

        # ===== 4) GammaLinearity 차트: 블록 중심 x (= g+4), dot+line =====
        centers = np.arange(88, 232, 8) + 4    # 92,100,...,228
        self.vac_optimization_gammalinearity_chart.set_series(
            "OFF_slope8", centers, m_off, marker='o', linestyle='-', label='OFF slope(8)'
        )
        self.vac_optimization_gammalinearity_chart.set_series(
            "ON_slope8",  centers, m_on,  marker='o', linestyle='--', label='ON slope(8)'
        )

        # ===== 5) ColorShift(4종) 표 & 6) 묶음 막대 =====
        # store['colorshift'][role]에는 op.colorshift_patterns 순서대로 (x,y,u′,v′)가 append되어 있음
        # 우리가 필요로 하는 4패턴 인덱스 찾기
        want_names = ['Dark Skin','Light Skin','Asian','Western']   # op 리스트의 라벨과 동일하게
        name_to_idx = {name: i for i, (name, *_rgb) in enumerate(op.colorshift_patterns)}

        def _delta_uv_for_state(state_store):
            # main=정면(0°), sub=측면(60°) 가정
            arr = []
            for nm in want_names:
                idx = name_to_idx.get(nm, None)
                if idx is None: arr.append(np.nan); continue
                if idx >= len(state_store['colorshift']['main']) or idx >= len(state_store['colorshift']['sub']):
                    arr.append(np.nan); continue
                _, _, u0, v0 = state_store['colorshift']['main'][idx]  # 정면
                _, _, u6, v6 = state_store['colorshift']['sub'][idx]   # 측면
                if not all(np.isfinite([u0,v0,u6,v6])):
                    arr.append(np.nan); continue
                d = float(np.sqrt((u6-u0)**2 + (v6-v0)**2))
                arr.append(d)
            return np.array(arr, dtype=np.float64)  # [DarkSkin, LightSkin, Asian, Western]

        duv_off = _delta_uv_for_state(off_store)
        duv_on  = _delta_uv_for_state(on_store)
        mean_off = float(np.nanmean(duv_off)) if np.isfinite(duv_off).any() else float('nan')
        mean_on  = float(np.nanmean(duv_on))  if np.isfinite(duv_on).any()  else float('nan')

        # 표 채우기: 2열=OFF, 3열=ON / 2~5행=패턴 / 6행=평균
        tbl_cs = self.ui.vac_table_colorShift_3
        # OFF
        _set_text(tbl_cs, 1, 1, f"{duv_off[0]:.6f}")   # DarkSkin
        _set_text(tbl_cs, 2, 1, f"{duv_off[1]:.6f}")   # LightSkin
        _set_text(tbl_cs, 3, 1, f"{duv_off[2]:.6f}")   # Asian
        _set_text(tbl_cs, 4, 1, f"{duv_off[3]:.6f}")   # Western
        _set_text(tbl_cs, 5, 1, f"{mean_off:.6f}")     # 평균
        # ON
        _set_text(tbl_cs, 1, 2, f"{duv_on[0]:.6f}")
        _set_text(tbl_cs, 2, 2, f"{duv_on[1]:.6f}")
        _set_text(tbl_cs, 3, 2, f"{duv_on[2]:.6f}")
        _set_text(tbl_cs, 4, 2, f"{duv_on[3]:.6f}")
        _set_text(tbl_cs, 5, 2, f"{mean_on:.6f}")

        # 묶음 막대 차트 갱신
        self.vac_optimization_colorshift_chart.update_grouped(
            data_off=list(np.nan_to_num(duv_off, nan=0.0)),
            data_on =list(np.nan_to_num(duv_on,  nan=0.0))
        )
