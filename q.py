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
        dG  = G_on - G_off
        dCx = cx_on - cx_off
        dCy = cy_on - cy_off

        def _pass_total_chroma(d_arr, thr):
            # 유효 값 + edge gray(0,1,254,255) 제외
            mask = np.isfinite(d_arr)
            for g in (0, 1, 254, 255):
                if 0 <= g < len(mask):
                    mask[g] = False
        
            vals = d_arr[mask]
            tot = int(np.sum(mask))
            if tot <= 0:
                return 0, 0
            
            rounded = np.round(np.abs(vals), 4)
            thr_r = round(float(thr), 4)
            ok = int(np.sum(rounded <= thr_r))
            return ok, tot
        
        def _pass_total_gamma(d_arr, thr):
            mask = np.isfinite(d_arr)
            for g in (0, 1, 254, 255):
                if 0 <= g < len(mask):
                    mask[g] = False

            vals = d_arr[mask]
            tot = int(np.sum(mask))
            if tot <= 0:
                return 0, 0
            
            rounded = np.round(np.abs(vals), 3)
            thr_r = round(float(thr), 3)
            ok = int(np.sum(rounded <= thr_r))
            return ok, tot

        ok_cx, tot_cx = _pass_total_chroma(dCx, thr_c)
        ok_cy, tot_cy = _pass_total_chroma(dCy, thr_c)
        ok_g , tot_g  = _pass_total_gamma(dG , thr_gamma)

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
