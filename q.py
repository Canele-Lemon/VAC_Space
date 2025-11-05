- 델타 Cx/Cy 스펙 소수점 4번째 자리에서 반올림 했을때 0.003이면 통과로 하기. 델타 Gamma 스펙도 마찬가지로 세번째자리에서 반올림했을때 0.05이면 통과한걸로 하기
- gray0,1,254,255는 평가에서 완전 제외하기 -> NG여도 셀 색깔 바꾸기없음. (default 그래도 두기) 그래서 _update_spec_views에서 평가결과 업데이트될때도 256-4=252개 중 몇개 통과로 바꾸어야 함 ->"통과개수/252"
- 현재 델타 Gamma평가는 델타 Cx,Cy와 달리 normalized 휘도 기준으로 계산되어야 하므로 전 gray 측정이 끝난 후 이루어지고 있음. 이를 다음과 같이 변경하여 델타 Gamma 평가도 즉석에서 이루어지게 하고자함:
VAC OFF에서 측정한 max 휘도와 VAC ON(또는 보정후)에서 측정한 0gray 휘도값 기준으로 normalized -> VAC ON(또는 보정후)측정할 때마다 Gamma 계산, 델타 Gamma 평가

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
        J = np.asarray(bundle["J"], dtype=np.float32) # (256,3,3)
        n = np.asarray(bundle["n"], dtype=np.int32)   # (256,)
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
            self._gamma_off_vec = self._compute_gamma_series(lv_off)
            
            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
            logging.info("[TV Control] VAC ON 전환 성공")
            
            logging.info("[Measurement] VAC ON 측정 시작")
            self._apply_vac_from_db_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off,
            gray_levels=op.gray_levels_256,
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            gamma_settle_ms=1000,
            cs_settle_ms=1000,
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
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")

            self._step_done(3)

            self._fine_mode = False
            
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_on = SessionProfile(
                legend_text="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=1))
                self._spec_thread.start()

            logging.info("[Measurement] DB fetch VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                gamma_settle_ms=1000,
                cs_settle_ms=1000,
                on_done=_after_on
            )

        logging.info("[VAC Writing] DB fetch VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(vac_data, on_finished=_after_write)
        
    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        """
        조건 1) spec_ok==True: 종료
        조건 2) (spec_ok==False) and (max_iters>0): Batch Correction
                └─ 조건 2-1) (len(ng_grays)<=10): Batch Correction 종료 → Per-gray Fine Correction 진입
                └─ 조건 2-2) (len(ng_grays)>10) and (iter_idx<max_iters): 다음 Batch Correction
        """
        try:
            ng_grays = []
            thr_g = None
            thr_c = None
            
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

            self._update_spec_views(iter_idx, self._off_store, self._on_store)

            # 조건 1) spec_ok==True: 종료
            if spec_ok:
                self._step_done(5)
                logging.info("[Evaluation] Spec 통과 — 최적화 종료")
                return
            
            self._step_fail(5)
            if max_iters <= 0:
                logging.info("[Evaluation] Spec NG but no further correction (max_iters≤0) - 최적화 종료")
                return
            
            # 조건 2) (spec_ok==False) and (max_iters>0): Batch Correction
            ng_cnt = len(ng_grays)
            # 조건 2-1) (len(ng_grays)<=10): Batch Correction 종료 → Per-gray Fine Correction 진입
            if ng_cnt > 0 and ng_cnt <= 10:
                logging.info(f"[Evaluation] NG gray {ng_cnt}개 ≤ 10 → Batch Correction 종료, Per-gray Fine Correction 시작")
                for s in (2, 3, 4):
                    self._step_set_pending(s)
                thr_gamma = float(thr_g) if thr_g is not None else 0.05
                thr_c_val = float(thr_c) if thr_c is not None else 0.003
                self._start_fine_correction_for_ng_list(
                    ng_grays,
                    thr_gamma=thr_gamma,
                    thr_c=thr_c_val
                )
                return
            # 조건 2-2) (len(ng_grays)>10) and (iter_idx<max_iters): 다음 Batch Correction
            if iter_idx < max_iters:
                logging.info(f"[Evaluation] Spec NG — batch 보정 {iter_idx+1}회차 시작")
                for s in (2, 3, 4):
                    self._step_set_pending(s)

                thr_gamma = float(thr_g) if thr_g is not None else 0.05
                thr_c_val = float(thr_c) if thr_c is not None else 0.003

                self._run_batch_correction_with_jacobian(
                    iter_idx=iter_idx+1,
                    max_iters=max_iters,
                    thr_gamma=thr_gamma,
                    thr_c=thr_c_val,
                    metrics=metrics
                )
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

    def _set_item_with_spec(self, table, row, col, value, *, is_spec_ok: bool):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))
        if is_spec_ok:
            item.setBackground(QColor(0, 0, 255))
        else:
            item.setBackground(QColor(255, 0, 0))

        table.scrollToItem(item, QAbstractItemView.PositionAtCenter)

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
            is_fine_mode = getattr(self, "_fine_mode", False)
            
            if is_on_session and is_fine_mode:
                ok_now = self._is_gray_spec_ok(gray, thr_gamma=0.05, thr_c=0.003, off_store=self._off_store, on_store=s['store'])
                
                if not ok_now and not self._sess.get('paused', False):
                    logging.info(f"[Fine Correction] gray={gray} NG → per-gray correction start")
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

    def start_viewing_angle_session(self,
        profile: SessionProfile,
        gray_levels=None,
        gamma_patterns=('white','red','green','blue'),
        colorshift_patterns=None,
        first_gray_delay_ms=3000,
        gamma_settle_ms=1000,
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
            'gamma_settle_ms': gamma_settle_ms,
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

            if s['g_idx'] == 0:
                delay = s['first_gray_delay_ms']
            else:
                delay = s.get('gamma_settle_ms', 0)
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

