    #┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    #│                                  - VAC Optimization Loop -                                   │
        self.ui.vac_btn_startOptimization.clicked.connect(self.start_VAC_optimization)
        self._vac_dict_cache = None

        
        self._off_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 
                                     'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                            'colorshift': {'main': [], 'sub': []}}
        self._on_store  = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 
                                     'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                            'colorshift': {'main': [], 'sub': []}}
        
        base = cf.get_normalized_path(__file__, '..', '..', 'resources/images/pictures')
        self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
        self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
        self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))
        
        self.vac_optimization_gamma_chart = GammaChart(self.ui.vac_chart_gamma_3)
        self.vac_optimization_cie1976_chart = CIE1976Chart(self.ui.vac_chart_colorShift_2)
        self.vac_optimization_lut_chart = LUTChart(target_widget=self.ui.vac_graph_rgbLUT_4)

        self.vac_optimization_chromaticity_chart = XYChart(
            target_widget=self.ui.vac_chart_chromaticityDiff,
            x_label='Gray Level', y_label='Cx/Cy',
            x_range=(0, 256), y_range=(0, 1),
            x_tick=64, y_tick=0.25,
            title=None, title_color='#595959',
            legend=True   # ← 변경
        )
        self.vac_optimization_gammalinearity_chart = XYChart(
            target_widget=self.ui.vac_chart_gammaLinearity,
            x_label='Gray Level',
            y_label='Slope',
            x_range=(0, 256),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title=None,
            title_color='#595959',
            legend=False
        )
        self.vac_optimization_colorshift_chart = BarChart(
            target_widget=self.ui.vac_chart_colorShift_3,
            title='Skin Color Shift',
            x_labels=['DarkSkin','LightSkin','Asian','Western'],
            y_label='Δu′v′',
            y_range=(0, 0.08), y_tick=0.02,
            series_labels=('VAC OFF','VAC ON'),
            spec_line=0.04
        )
        
    def _load_jacobian_bundle_npy(self):
        """
        bundle["J"]   : (256,3,3)
        bundle["n"]   : (256,)
        bundle["cond"]: (256,)
        """
        jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_bundle_ref2582_lam0.001_dw50.0_gs30.0_20251104_092159.npy')  # 파일명은 실제꺼로 수정
        if not os.path.exists(jac_path):
            logging.error(f"[Jacobian] npy 파일을 찾을 수 없습니다: {jac_path}")
            raise FileNotFoundError(f"Jacobian npy not found: {jac_path}")

        bundle = np.load(jac_path, allow_pickle=True).item()
        J = np.asarray(bundle["J"], dtype=np.float32)      # (256,3,3)
        n = np.asarray(bundle["n"], dtype=np.int32)        # (256,)
        cond = np.asarray(bundle["cond"], dtype=np.float32)

        self._jac_bundle = bundle
        self._J_dense = J
        self._J_n = n
        self._J_cond = cond

        logging.info(f"[Jacobian] dense J bundle loaded: {jac_path}, J.shape={J.shape}")

    def _run_off_baseline_then_on(self):
        profile_off = SessionProfile(
            legend_text="VAC OFF (Ref.)",
            cie_label="data_1",
            table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
            ref_store=None
        )

        def _after_off(store_off):
            self._off_store = store_off
            lv_off = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = store_off['gamma']['main']['white'].get(g, None)
                lv_off[g] = float(tup[0]) if tup else np.nan
                
            self._off_lv_vec = lv_off
            self._off_lv0 = float(lv_off[0])
            
            with np.errstate(invalid='ignore'):
                self._off_denom = float(np.nanmax(lv_off[1:] - self._off_lv0)) if np.isfinite(self._off_lv0) else np.nan
            
            self._gamma_off_vec = self._compute_gamma_series(lv_off)

            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
                
            self._apply_vac_from_db_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off, 
            gray_levels=op.gray_levels_256, 
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_off
        )
    
    def _apply_vac_from_db_and_measure_on(self):
        self._step_start(2)
        
        # panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        # fr = self.ui.vac_cmb_FrameRate.currentText().strip()
        # vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        # if vac_data is None:
        #     logging.error(f"[DB] {panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
        #     return

        vac_version, vac_data = self._fetch_vac_by_vac_info_pk(2582)
        if vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        vac_dict = json.loads(vac_data)
        self._vac_dict_cache = vac_dict
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in vac_dict.items() if "channel" in key}
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] DB fetch VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] DB fetch VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=vac_dict, read_data=read_vac_dict)

            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] VAC 데이터 일치")

            self._step_done(3)

            # Gamma / Color Shift 차트 "ON" 시리즈 Reset
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_on = SessionProfile(
                legend_text="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                self._step_done(4)
                self._on_store = store_on
                
                self._step_start(5)
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=2))
                self._spec_thread.start()

            self._step_start(4)
            logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 시작")
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000, cs_settle_ms=1000,
                on_done=_after_on
            )

        logging.info("[VAC Writing] DB fetch VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(vac_data, on_finished=_after_write)
        
    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        try:
            if metrics and "error" not in metrics:
                max_dG   = metrics.get("max_dG",  float("nan"))
                max_dCx  = metrics.get("max_dCx", float("nan"))
                max_dCy  = metrics.get("max_dCy", float("nan"))
                thr_g    = metrics.get("thr_gamma", self._spec_thread.thr_gamma if self._spec_thread else None)
                thr_c    = metrics.get("thr_c",     self._spec_thread.thr_c     if self._spec_thread else None)
                ng_grays = metrics.get("ng_grays", [])
                logging.info(
                    f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{thr_g}), "
                    f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c}), "
                    f"NG grays={ng_grays}"
                )
            else:
                logging.warning("[Evaluation] evaluation failed — treating as not passed.")
                ng_grays = []

            # 결과 표/차트 갱신
            self._update_spec_views(iter_idx, self._off_store, self._on_store)

            if spec_ok:
                self._step_done(5)
                logging.info("[Evaluation] Spec 통과 — 최적화 종료")
                return

            self._step_fail(5)
            if iter_idx < max_iters:
                logging.info(f"[Evaluation] Spec NG — Spec NG — 보정 {iter_idx+1}회차 시작")
                for s in (2,3,4):
                    self._step_set_pending(s)
                self._run_batch_correction_with_jacobian(iter_idx=iter_idx+1, max_iters=max_iters)
            else:
                logging.info("[Correction] 최대 보정 횟수 도달 — 종료")
        finally:
            self._spec_thread = None
        
    def _update_spec_views(self, iter_idx, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
        """
        결과 표/차트 갱신
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
        
        logging.debug(f"{iter_idx}차 보정 결과: Cx:{ok_cx}/{tot_cx}, Cy:{ok_cy}/{tot_cy}, Gamma:{ok_g}/{tot_g}")

        # ===== 2) ChromaticityDiff 차트: Cx/Cy vs gray (OFF/ON) =====
        x = np.arange(256)
        # 1) 먼저 데이터 넣기 (색/스타일 우리가 직접 세팅)
        self.vac_optimization_chromaticity_chart.set_series(
            "OFF_Cx", x, cx_off,
            marker=None,
            linestyle='--',
            label='OFF Cx'
        )
        self.vac_optimization_chromaticity_chart.lines["OFF_Cx"].set_color('orange')

        self.vac_optimization_chromaticity_chart.set_series(
            "ON_Cx", x, cx_on,
            marker=None,
            linestyle='-',
            label='ON Cx'
        )
        self.vac_optimization_chromaticity_chart.lines["ON_Cx"].set_color('orange')

        self.vac_optimization_chromaticity_chart.set_series(
            "OFF_Cy", x, cy_off,
            marker=None,
            linestyle='--',
            label='OFF Cy'
        )
        self.vac_optimization_chromaticity_chart.lines["OFF_Cy"].set_color('green')

        self.vac_optimization_chromaticity_chart.set_series(
            "ON_Cy", x, cy_on,
            marker=None,
            linestyle='-',
            label='ON Cy'
        )
        self.vac_optimization_chromaticity_chart.lines["ON_Cy"].set_color('green')
        
        # y축 autoscale with margin 1.1
        all_y = np.concatenate([
            np.asarray(cx_off, dtype=np.float64),
            np.asarray(cx_on,  dtype=np.float64),
            np.asarray(cy_off, dtype=np.float64),
            np.asarray(cy_on,  dtype=np.float64),
        ])
        all_y = all_y[np.isfinite(all_y)]
        if all_y.size > 0:
            ymin = np.min(all_y)
            ymax = np.max(all_y)
            center = 0.5*(ymin+ymax)
            half = 0.5*(ymax-ymin)
            # half==0일 수도 있으니 최소폭을 조금 만들어주자
            if half <= 0:
                half = max(0.001, abs(center)*0.05)
            half *= 1.1  # 10% margin
            new_min = center - half
            new_max = center + half

            ax_chr = self.vac_optimization_chromaticity_chart.ax
            cs.MatFormat_Axis(ax_chr, min_val=np.float64(new_min),
                                        max_val=np.float64(new_max),
                                        tick_interval=None,
                                        axis='y')
            ax_chr.relim(); ax_chr.autoscale_view(scalex=False, scaley=False)
            self.vac_optimization_chromaticity_chart.canvas.draw()

        # ===== 3) GammaLinearity 표: 88~232, 8gray 블록 평균 슬로프 =====
        def _normalized_luminance(lv_vec):
            """
            lv_vec: (256,) 절대 휘도 [cd/m2]
            return: (256,) 0~1 정규화된 휘도
                    Ynorm[g] = (Lv[g] - Lv[0]) / (max(Lv[1:]-Lv[0]))
            감마 계산과 동일한 노말라이제이션 방식 유지
            """
            lv_arr = np.asarray(lv_vec, dtype=np.float64)
            y0 = lv_arr[0]
            denom = np.nanmax(lv_arr[1:] - y0)
            if not np.isfinite(denom) or denom <= 0:
                return np.full(256, np.nan, dtype=np.float64)
            return (lv_arr - y0) / denom

        def _block_slopes(lv_vec, g_start=88, g_stop=232, step=8):
            """
            lv_vec: (256,) 절대 휘도
            g_start..g_stop: 마지막 블록은 [224,232]까지 포함되도록 설정
            step: 8gray 폭

            return:
            mids  : (n_blocks,) 각 블록 중간 gray (예: 92,100,...,228)
            slopes: (n_blocks,) 각 블록의 slope
                    slope = abs( Ynorm[g1] - Ynorm[g0] ) / ((g1-g0)/255)
                    g0 = block start, g1 = block end (= g0+step)
            """
            Ynorm = _normalized_luminance(lv_vec)  # (256,)
            mids   = []
            slopes = []
            for g0 in range(g_start, g_stop, step):
                g1 = g0 + step
                if g1 >= len(Ynorm):
                    break

                y0 = Ynorm[g0]
                y1 = Ynorm[g1]

                # 분모 = gray step을 0~1로 환산한 Δgray_norm
                d_gray_norm = (g1 - g0) / 255.0

                if np.isfinite(y0) and np.isfinite(y1) and d_gray_norm > 0:
                    slope = abs(y1 - y0) / d_gray_norm
                else:
                    slope = np.nan

                mids.append(g0 + (g1 - g0)/2.0)  # 예: 88~96 -> 92.0
                slopes.append(slope)

            return np.asarray(mids, dtype=np.float64), np.asarray(slopes, dtype=np.float64)

        mids_off, slopes_off = _block_slopes(lv_off, g_start=88, g_stop=232, step=8)
        mids_on , slopes_on  = _block_slopes(lv_on , g_start=88, g_stop=232, step=8)

        avg_off = float(np.nanmean(slopes_off)) if np.isfinite(slopes_off).any() else float('nan')
        avg_on  = float(np.nanmean(slopes_on )) if np.isfinite(slopes_on ).any() else float('nan')

        tbl_gl = self.ui.vac_table_gammaLinearity
        _set_text(tbl_gl, 1, 1, f"{avg_off:.6f}")  # 2행,2열 OFF 평균 기울기
        _set_text(tbl_gl, 1, 2, f"{avg_on:.6f}")   # 2행,3열 ON  평균 기울기

        # ===== 4) GammaLinearity 차트: 블록 중심 x (= g+4), dot+line =====
        # 라인 세팅
        self.vac_optimization_gammalinearity_chart.set_series(
            "OFF_slope8",
            mids_off,
            slopes_off,
            marker='o',
            linestyle='-',
            label='OFF slope(8)'
        )
        off_ln = self.vac_optimization_gammalinearity_chart.lines["OFF_slope8"]
        off_ln.set_color('black')
        off_ln.set_markersize(3)   # 기존보다 작게 (기본이 6~8 정도일 가능성)

        self.vac_optimization_gammalinearity_chart.set_series(
            "ON_slope8",
            mids_on,
            slopes_on,
            marker='o',
            linestyle='-',
            label='ON slope(8)'
        )
        on_ln = self.vac_optimization_gammalinearity_chart.lines["ON_slope8"]
        on_ln.set_color('red')
        on_ln.set_markersize(3)

        # y축 autoscale with margin 1.1
        all_slopes = np.concatenate([
            np.asarray(slopes_off, dtype=np.float64),
            np.asarray(slopes_on,  dtype=np.float64),
        ])
        all_slopes = all_slopes[np.isfinite(all_slopes)]
        if all_slopes.size > 0:
            ymin = np.min(all_slopes)
            ymax = np.max(all_slopes)
            center = 0.5*(ymin+ymax)
            half = 0.5*(ymax-ymin)
            if half <= 0:
                half = max(0.001, abs(center)*0.05)
            half *= 1.1  # 10% margin
            new_min = center - half
            new_max = center + half

            ax_slope = self.vac_optimization_gammalinearity_chart.ax
            cs.MatFormat_Axis(ax_slope,
                            min_val=np.float64(new_min),
                            max_val=np.float64(new_max),
                            tick_interval=None,
                            axis='y')
            ax_slope.relim(); ax_slope.autoscale_view(scalex=False, scaley=False)
            self.vac_optimization_gammalinearity_chart.canvas.draw()

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
                if idx is None: 
                    arr.append(np.nan)
                    continue
                if idx >= len(state_store['colorshift']['main']) or idx >= len(state_store['colorshift']['sub']):
                    arr.append(np.nan)
                    continue
                lv0, u0, v0 = state_store['colorshift']['main'][idx]  # 정면
                lv6, u6, v6 = state_store['colorshift']['sub'][idx]   # 측면
                
                if not all(np.isfinite([u0, v0, u6, v6])):
                    arr.append(np.nan)
                    continue
                
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

    def _run_batch_correction_with_jacobian(self, iter_idx, max_iters, thr_gamma, thr_c, lam=1e-3, metrics=None):

        logging.info(f"[Correction] iteration {iter_idx} start (Jacobian dense)")

        # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
        if not hasattr(self, "_J_dense"):
            logging.error("[Correction] J_dense not loaded") # self._J_dense 없음
            return
        self._load_mapping_index_gray_to_lut()
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.error("[Correction] no VAC cache; need latest TV VAC JSON")
            return

        # 1) NG gray 리스트 / Δ 타깃 준비
        if metrics is not None and "ng_grays" in metrics and "dG" in metrics:
            ng_list = list(metrics["ng_grays"])
            d_targets = {
                "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
                "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
                "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
            }
            thr_gamma = float(metrics.get("thr_gamma", thr_gamma))
            thr_c     = float(metrics.get("thr_c",     thr_c))
            logging.info(f"[Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
        else:
            dG, dCx, dCy, ng_list = SpecEvalThread.compute_gray_errors_and_ng_list(
                self._off_store, self._on_store,
                thr_gamma=thr_gamma, thr_c=thr_c
            )
            d_targets = {
                "Gamma": dG.astype(np.float32),
                "Cx":    dCx.astype(np.float32),
                "Cy":    dCy.astype(np.float32),
            }
            logging.info(f"[Correction] NG grays (recomputed): {ng_list}")

        if not ng_list:
            logging.info("[BATCH CORR] no NG gray (또는 0/1/254/255만 NG) → 보정 없음")
            return
    
        vac_dict = self._vac_dict_cache
    
        # 2) 현재 High LUT 확보
        vac_dict = self._vac_dict_cache

        RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
        GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
        BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

        RH = RH0.copy()
        GH = GH0.copy()
        BH = BH0.copy()

        # 3) index별 Δ 누적 (여러 gray가 같은 index를 참조할 수 있으므로)
        delta_acc = {
            "R": np.zeros_like(RH),
            "G": np.zeros_like(GH),
            "B": np.zeros_like(BH),
        }
        count_acc = {
            "R": np.zeros_like(RH, dtype=np.int32),
            "G": np.zeros_like(GH, dtype=np.int32),
            "B": np.zeros_like(BH, dtype=np.int32),
        }

        mapR = self._lut_map_high["R"]   # (256,)
        mapG = self._lut_map_high["G"]
        mapB = self._lut_map_high["B"]
        
        # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
        for g in ng_list:
            dX = self._solve_delta_rgb_for_gray(
                g,
                d_targets,
                lam=lam,
                wCx=0.5,
                wCy=0.5,
                wG=1.0,
            )
            if dX is None:
                continue

            dR, dG, dB = dX

            idxR = int(mapR[g])
            idxG = int(mapG[g])
            idxB = int(mapB[g])

            if 0 <= idxR < len(RH):
                delta_acc["R"][idxR] += dR
                count_acc["R"][idxR] += 1
            if 0 <= idxG < len(GH):
                delta_acc["G"][idxG] += dG
                count_acc["G"][idxG] += 1
            if 0 <= idxB < len(BH):
                delta_acc["B"][idxB] += dB
                count_acc["B"][idxB] += 1

        # 5) index별 평균 Δ 적용 + clip + monotone + 로그
        for ch, arr, arr0 in (
            ("R", RH, RH0),
            ("G", GH, GH0),
            ("B", BH, BH0),
        ):
            da = delta_acc[ch]
            ct = count_acc[ch]
            mask = ct > 0

            if not np.any(mask):
                logging.info(f"[BATCH CORR] channel {ch}: no indices updated")
                continue

            # 평균 Δ
            arr[mask] = arr0[mask] + (da[mask] / ct[mask])
            # clip
            arr[:] = np.clip(arr, 0.0, 4095.0)
            # 단조 증가 (i<j → LUT[i] ≤ LUT[j])
            self._enforce_monotone(arr)

            # 🔹 인덱스별 보정 로그 (before → after)
            changed_idx = np.where(mask)[0]
            logging.info(f"[BATCH CORR] channel {ch}: {len(changed_idx)} indices updated")
            for idx in changed_idx:
                before = float(arr0[idx])
                after  = float(arr[idx])
                delta  = after - before
                logging.debug(
                    f"[BATCH CORR] ch={ch} idx={idx:4d}: {before:7.1f} → {after:7.1f} (Δ={delta:+.2f})"
                )

        # 6) NG gray 기준으로 어떤 LUT index가 어떻게 바뀌었는지 추가 요약 로그
        for g in ng_list:
            idxR = int(mapR[g])
            idxG = int(mapG[g])
            idxB = int(mapB[g])
            info = []
            if 0 <= idxR < len(RH0):
                info.append(
                    f"R(idx={idxR}): {RH0[idxR]:.1f}→{RH[idxR]:.1f} (Δ={RH[idxR]-RH0[idxR]:+.1f})"
                )
            if 0 <= idxG < len(GH0):
                info.append(
                    f"G(idx={idxG}): {GH0[idxG]:.1f}→{GH[idxG]:.1f} (Δ={GH[idxG]-GH0[idxG]:+.1f})"
                )
            if 0 <= idxB < len(BH0):
                info.append(
                    f"B(idx={idxB}): {BH0[idxB]:.1f}→{BH[idxB]:.1f} (Δ={BH[idxB]-BH0[idxB]:+.1f})"
                )
            if info:
                logging.info(f"[BATCH CORR] g={g:3d} → " + " | ".join(info))

        # 7) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "RchannelHigh": RH,
            "GchannelHigh": GH,
            "BchannelHigh": BH,
        }
        for k in new_lut_4096:
            new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

        # UI용 플롯 dict
        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],
            "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],
            "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],
            "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)

        # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
        logging.info(f"[Correction] LUT {iter_idx}차 보정 완료")

        vac_write_json = self.build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )

        def _after_write(ok, msg):
            logging.info(f"[VAC Writing] write result: {ok} {msg}")
            if not ok:
                return
            logging.info("[BATCH CORR] TV reading after write")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            if not vac_dict_after:
                logging.error("[BATCH CORR] TV read-back failed")
                return
            self._vac_dict_cache = vac_dict_after
            self._step_done(3)

            # ON 시리즈 리셋
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_corr = SessionProfile(
                legend_text=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7,
                            "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr
                self._step_start(5)
                self._spec_thread = SpecEvalThread(
                    self._off_store, self._on_store,
                    thr_gamma=thr_gamma, thr_c=thr_c, parent=self
                )
                self._spec_thread.finished.connect(
                    lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx, max_iters)
                )
                self._spec_thread.start()

            logging.info("[BATCH CORR] re-measure start (after LUT update)")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_corr,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000, cs_settle_ms=1000,
                on_done=_after_corr
            )

        self._step_start(3)
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)
        
    def _solve_delta_rgb_for_gray(
        self,
        g: int,
        d_targets: dict,
        lam: float = 1e-3,
        wCx: float = 0.5,
        wCy: float = 0.5,
        wG:  float = 1.0,
    ):
        """
        주어진 gray g에서, 현재 ΔY = [dCx, dCy, dGamma]를
        자코비안 J_g를 이용해 줄이기 위한 ΔX = [ΔR_H, ΔG_H, ΔB_H]를 푼다.

        관계식:  ΔY_new ≈ ΔY + J_g · ΔX
        우리가 원하는 건 ΔY_new ≈ 0 이므로, J_g · ΔX ≈ -ΔY 를 풀어야 함.

        리지 가중 최소자승:
            argmin_ΔX || W (J_g ΔX + ΔY) ||^2 + λ ||ΔX||^2
            → (J^T W^2 J + λI) ΔX = - J^T W^2 ΔY
        """
        Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3,3)
        if not np.isfinite(Jg).all():
            logging.warning(f"[BATCH CORR] g={g}: J_g has NaN/inf → skip")
            return None

        dCx_g = float(d_targets["Cx"][g])
        dCy_g = float(d_targets["Cy"][g])
        dG_g  = float(d_targets["Gamma"][g])
        dy = np.array([dCx_g, dCy_g, dG_g], dtype=np.float32)  # (3,)

        # 이미 거의 0이면 굳이 보정 안 해도 됨
        if np.all(np.abs(dy) < 1e-6):
            return None

        # 가중치
        w_vec = np.array([wCx, wCy, wG], dtype=np.float32)     # (3,)
        WJ = w_vec[:, None] * Jg   # (3,3)
        Wy = w_vec * dy            # (3,)

        A = WJ.T @ WJ + float(lam) * np.eye(3, dtype=np.float32)  # (3,3)
        b = - WJ.T @ Wy                                           # (3,)

        try:
            dX = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

        dR, dG, dB = float(dX[0]), float(dX[1]), float(dX[2])
        logging.debug(
            f"[BATCH CORR] g={g}: dCx={dCx_g:+.6f}, dCy={dCy_g:+.6f}, dG={dG_g:+.6f} → "
            f"ΔR_H={dR:+.3f}, ΔG_H={dG:+.3f}, ΔB_H={dB:+.3f}"
        )
        return dR, dG, dB

    def start_viewing_angle_session(self,
        profile: SessionProfile,
        gray_levels=None,
        gamma_patterns=('white','red','green','blue'),
        colorshift_patterns=None,
        first_gray_delay_ms=3000,
        cs_settle_ms=1000,
        on_done=None,
    ):
        if gray_levels is None:
            gray_levels = op.gray_levels_256
        if colorshift_patterns is None:
            colorshift_patterns = op.colorshift_patterns
        
        gamma_patterns=('white',)
        store = {
            'gamma': {'main': {p:{} for p in gamma_patterns}, 'sub': {p:{} for p in gamma_patterns}},
            'colorshift': {'main': [], 'sub': []}
        }

        self._sess = {
            'phase': 'gamma',
            'p_idx': 0,
            'g_idx': 0,
            'cs_idx': 0,
            'patterns': list(gamma_patterns),
            'gray_levels': list(gray_levels),
            'cs_patterns': colorshift_patterns,
            'store': store,
            'profile': profile,
            'first_gray_delay_ms': first_gray_delay_ms,
            'cs_settle_ms': cs_settle_ms,
            'on_done': on_done
        }
        self._session_step()

    def _session_step(self):
        s = self._sess
        if s.get('paused', False):
            return
        
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
            got_sub = ('sub') in s['_gamma'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_gamma_pair(pattern, gray, s['_gamma'])
                
                if s.get('paused', False):
                    return
                
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
        """
        results: {
        'main': (x, y, lv, cct, duv)  또는  None,
        'sub' : (x, y, lv, cct, duv)  또는  None
        }
        """
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
                continue

            x, y, lv, cct, duv = res
            store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

            self.vac_optimization_gamma_chart.add_point(
                state=state,
                role=role,               # 'main'/'sub'
                pattern=pattern,         # 'white'/'red'/'green'/'blue'
                gray=int(gray),
                luminance=float(lv)
            )

        if pattern == 'white':
            is_on_session = (profile.ref_store is not None)
            if is_on_session:
                ok_now = self._is_gray_spec_ok(gray, thr_gamma=0.05, thr_c=0.003, off_store=self._off_store, on_store=s['store'])
                if not ok_now and not self._sess.get('paused', False):
                    self._start_gray_ng_correction(gray, max_retries=3, thr_gamma=0.05, thr_c=0.003)
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
                if ref_main is not None and np.isfinite(cx_m) and np.isfinite(cy_m):
                    _, cx_r, cy_r = ref_main
                    d_cx = cx_m - cx_r
                    d_cy = cy_m - cy_r
                    self._set_item_with_spec(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}", is_spec_ok=(abs(d_cx) <= 0.003))
                    self._set_item_with_spec(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}", is_spec_ok=(abs(d_cy) <= 0.003))

    def _trigger_colorshift_pair(self, patch_name):
        s = self._sess
        s['_cs'] = {}

        def handle(role, res):
            s['_cs'][role] = res
            got_main = 'main' in s['_cs']
            got_sub = ('sub') in s['_cs'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_colorshift_pair(patch_name, s['_cs'])
                s['cs_idx'] += 1
                QTimer.singleShot(80, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle)
            self.sub_measure_thread.start()

    def _consume_colorshift_pair(self, patch_name, results):
        """
        results: {
            'main': (x, y, lv, cct, duv)  또는  None,   # main = 0°
            'sub' : (x, y, lv, cct, duv)  또는  None    # sub  = 60°
        }
        """
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        # 현재 세션 상태 문자열 ('VAC OFF...' 이면 OFF, 아니면 ON)
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        # 이 측정 패턴의 row index (op.colorshift_patterns 순서 그대로)
        row_idx = s['cs_idx']

        # 이 테이블: vac_table_opt_mes_results_colorshift
        tbl_cs_raw = self.ui.vac_table_opt_mes_results_colorshift

        # ------------------------------------------------
        # 1) main / sub 결과 변환해서 store에 넣고 차트 갱신
        #    store['colorshift'][role][row_idx] = (Lv, u', v')
        # ------------------------------------------------
        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                # 측정 실패 시 해당 row에 placeholder 저장
                store['colorshift'][role].append((np.nan, np.nan, np.nan))
                continue

            x, y, lv, cct, duv_unused = res

            # xy -> u' v'
            u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))

            # store에 (Lv, u', v') 저장
            store['colorshift'][role].append((
                float(lv),
                float(u_p),
                float(v_p),
            ))

            # 차트 갱신 (vac_optimization_cie1976_chart 는 u' v' scatter)
            self.vac_optimization_cie1976_chart.add_point(
                state=state,
                role=role,      # 'main' or 'sub'
                u_p=float(u_p),
                v_p=float(v_p)
            )

        # ------------------------------------------------
        # 2) 표 업데이트
        #    OFF 세션:
        #        2열,3열,4열 ← main의 Lv / u' / v'
        #    ON/CORR 세션:
        #        5열,6열,7열 ← main의 Lv / u' / v'
        #        8열        ← du'v' (sub vs main 거리)
        # ------------------------------------------------

        # 이제 방금 append한 값들을 row_idx에서 꺼냄
        main_ok = row_idx < len(store['colorshift']['main'])
        sub_ok  = row_idx < len(store['colorshift']['sub'])

        if main_ok:
            lv_main, up_main, vp_main = store['colorshift']['main'][row_idx]
        else:
            lv_main, up_main, vp_main = (np.nan, np.nan, np.nan)

        if sub_ok:
            lv_sub, up_sub, vp_sub = store['colorshift']['sub'][row_idx]
        else:
            lv_sub, up_sub, vp_sub = (np.nan, np.nan, np.nan)

        # 테이블에 안전하게 set 하는 helper
        def _safe_set_item(table, r, c, text):
            self._set_item(table, r, c, text if text is not None else "")

        if profile.legend_text.startswith('VAC OFF'):
            # ---------- VAC OFF ----------
            # row_idx 행의
            #   col=1 → Lv(main)
            #   col=2 → u'(main)
            #   col=3 → v'(main)

            txt_lv_off = f"{lv_main:.6f}" if np.isfinite(lv_main) else ""
            txt_u_off  = f"{up_main:.6f}"  if np.isfinite(up_main)  else ""
            txt_v_off  = f"{vp_main:.6f}"  if np.isfinite(vp_main)  else ""

            _safe_set_item(tbl_cs_raw, row_idx, 1, txt_lv_off)
            _safe_set_item(tbl_cs_raw, row_idx, 2, txt_u_off)
            _safe_set_item(tbl_cs_raw, row_idx, 3, txt_v_off)

        else:
            # ---------- VAC ON (또는 CORR 이후) ----------
            # row_idx 행의
            #   col=4 → Lv(main)
            #   col=5 → u'(main)
            #   col=6 → v'(main)
            #   col=7 → du'v' = sqrt((u'_sub - u'_main)^2 + (v'_sub - v'_main)^2)

            txt_lv_on = f"{lv_main:.6f}" if np.isfinite(lv_main) else ""
            txt_u_on  = f"{up_main:.6f}"  if np.isfinite(up_main)  else ""
            txt_v_on  = f"{vp_main:.6f}"  if np.isfinite(vp_main)  else ""

            _safe_set_item(tbl_cs_raw, row_idx, 4, txt_lv_on)
            _safe_set_item(tbl_cs_raw, row_idx, 5, txt_u_on)
            _safe_set_item(tbl_cs_raw, row_idx, 6, txt_v_on)

            # du'v' 계산
            # 엑셀식: =SQRT( (60deg_u' - 0deg_u')^2 + (60deg_v' - 0deg_v')^2 )
            # 여기서 main=0°, sub=60°
            duv_txt = ""
            if np.isfinite(up_main) and np.isfinite(vp_main) and np.isfinite(up_sub) and np.isfinite(vp_sub):
                dist = np.sqrt((up_sub - up_main)**2 + (vp_sub - vp_main)**2)
                duv_txt = f"{dist:.6f}"

            _safe_set_item(tbl_cs_raw, row_idx, 7, duv_txt)
        
    def _finalize_session(self):
        s = self._sess
        profile: SessionProfile = s['profile']
        table_main = self.ui.vac_table_opt_mes_results_main
        cols = profile.table_cols
        thr_gamma = 0.05

        # =========================
        # 1) main 감마 컬럼 채우기
        # =========================
        lv_series_main = np.zeros(256, dtype=np.float64)
        for g in range(256):
            tup = s['store']['gamma']['main']['white'].get(g, None)
            lv_series_main[g] = float(tup[0]) if tup else np.nan

        gamma_vec = self._compute_gamma_series(lv_series_main)
        for g in range(256):
            if np.isfinite(gamma_vec[g]):
                self._set_item(table_main, g, cols['gamma'], f"{gamma_vec[g]:.6f}")

        # =========================
        # 2) ΔGamma (ON세션일 때만)
        # =========================
        if profile.ref_store is not None and 'd_gamma' in cols:
            ref_lv_main = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = profile.ref_store['gamma']['main']['white'].get(g, None)
                ref_lv_main[g] = float(tup[0]) if tup else np.nan
            ref_gamma = self._compute_gamma_series(ref_lv_main)
            dG = gamma_vec - ref_gamma
            for g in range(256):
                if np.isfinite(dG[g]):
                    self._set_item_with_spec(
                        table_main, g, cols['d_gamma'], f"{dG[g]:.6f}",
                        is_spec_ok=(abs(dG[g]) <= thr_gamma)
                    )

        # =================================================================
        # 3) [ADD: slope 계산 후 sub 테이블 업데이트 - 측정 종료 후 한 번에]
        # =================================================================
        # 요구사항:
        # - sub 측정 white의 lv로 normalized 휘도 계산
        # - 88gray부터 8 gray step씩 (88→96, 96→104, ... 224→232)
        # - slope = abs( Ynorm[g+8] - Ynorm[g] ) / ((8)/255)
        # - slope는 row=g 에 기록
        # - VAC OFF 세션이면 sub 테이블의 4번째 열(0-based index 3)
        #   VAC ON / CORR 세션이면 sub 테이블의 8번째 열(0-based index 7)

        table_sub = self.ui.vac_table_opt_mes_results_sub

        # 3-1) sub white lv 배열 뽑기
        lv_series_sub = np.full(256, np.nan, dtype=np.float64)
        for g in range(256):
            tup_sub = s['store']['gamma']['sub']['white'].get(g, None)
            if tup_sub:
                lv_series_sub[g] = float(tup_sub[0])

        # 3-2) 정규화된 휘도 Ynorm[g] = (Lv[g]-Lv[0]) / max(Lv[1:]-Lv[0])
        def _norm_lv(lv_arr):
            lv0 = lv_arr[0]
            denom = np.nanmax(lv_arr[1:] - lv0)
            if not np.isfinite(denom) or denom <= 0:
                return np.full_like(lv_arr, np.nan, dtype=np.float64)
            return (lv_arr - lv0) / denom

        Ynorm_sub = _norm_lv(lv_series_sub)

        # 3-3) 어느 열에 쓰는지 결정
        is_off_session = profile.legend_text.startswith('VAC OFF')
        slope_col_idx = 3 if is_off_session else 7  # 4번째 or 8번째 열

        # 3-4) 각 8gray 블록 slope 계산해서 테이블에 기록
        # 블록 시작 gray: 88,96,104,...,224
        for g0 in range(88, 225, 8):
            g1 = g0 + 8
            if g1 >= 256:
                break

            y0 = Ynorm_sub[g0]
            y1 = Ynorm_sub[g1]
            d_gray_norm = (g1 - g0) / 255.0  # 8/255

            if np.isfinite(y0) and np.isfinite(y1) and d_gray_norm > 0:
                slope_val = abs(y1 - y0) / d_gray_norm
                txt = f"{slope_val:.6f}"
            else:
                txt = ""

            # row = g0 에 기록
            self._set_item(table_sub, g0, slope_col_idx, txt)

        # 끝났으면 on_done 콜백 실행
        if callable(s['on_done']):
            try:
                s['on_done'](s['store'])
            except Exception as e:
                logging.exception(e)
                    
    def _is_gray_spec_ok(self, gray:int, *, thr_gamma=0.05, thr_c=0.003,
                        off_store=None, on_store=None) -> bool:
        off_store = off_store if off_store is not None else self._off_store
        on_store  = on_store  if on_store  is not None else self._on_store
        ref = off_store['gamma']['main']['white'].get(gray, None)
        on  = on_store ['gamma']['main']['white'].get(gray, None)
        if not ref or not on:
            return True
        lv_r, cx_r, cy_r = ref
        lv_o, cx_o, cy_o = on

        dCx = abs(cx_o - cx_r) if (np.isfinite(cx_o) and np.isfinite(cx_r)) else 0.0
        dCy = abs(cy_o - cy_r) if (np.isfinite(cy_o) and np.isfinite(cy_r)) else 0.0

        # Gamma(OFF 정규화 프록시)
        if hasattr(self, "_gamma_off_vec") and hasattr(self, "_off_lv_vec"):
            G_ref_g = float(self._gamma_off_vec[gray])
            G_on_g  = self._gamma_from_off_norm_at_gray(self._off_lv_vec, lv_on_g=lv_o, g=gray)
            dG = abs(G_on_g - G_ref_g) if (np.isfinite(G_on_g) and np.isfinite(G_ref_g)) else 0.0
        else:
            dG = 0.0

        return (dCx <= thr_c) and (dCy <= thr_c) and (dG <= thr_gamma)
        
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

    def _finish_gray_fix(self, gray:int, *, pass_now: bool):
        ctx = self._sess.get('_gray_fix', None)
        if not ctx:
            self._resume_session(); return
        if pass_now or ctx['tries'] >= ctx['max']:
            logging.info(f"[GRAY-FIX] g={gray} {'PASS' if pass_now else 'MAX RETRIES'} → resume")
            self._sess['_gray_fix'] = None
            self._resume_session()
        else:
            self._do_gray_fix_once()  # 다음 재시도

    def _remeasure_same_gray(self, gray:int):
        """paused 상태에서 같은 g만 다시 측정 → store 반영 → 그 자리에서 PASS 판정"""
        s = self._sess
        self.changeColor(f"{gray},{gray},{gray}")
        payload = {}

        def handle(role, res):
            payload[role] = res
            got_main = ('main' in payload)
            got_sub  = ('sub' in payload) or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                # 기존 소비 로직 재사용(차트/테이블 업데이트)
                self._consume_gamma_pair('white', gray, payload)
                ok = self._is_gray_spec_ok(gray, off_store=self._off_store, on_store=s['store'])
                self._finish_gray_fix(gray, pass_now=ok)

        if self.main_instrument_cls:
            t1 = MeasureThread(self.main_instrument_cls, 'main')
            t1.measure_completed.connect(handle); t1.start()
        if self.sub_instrument_cls:
            t2 = MeasureThread(self.sub_instrument_cls, 'sub')
            t2.measure_completed.connect(handle); t2.start()

    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        """
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        try:
            self._load_jacobian_bundle_npy()
        except Exception as e:
            logging.exception("[Jacobian] Jacobian load failed")
            return
        
        # 1.2 TV VAC OFF 하기
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")    
        
        # 1.3 OFF 측정 세션 시작
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()
