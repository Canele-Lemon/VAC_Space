    def _consume_gamma_pair(self, pattern, gray, results):
        COLOR_EVAL_EXCLUDED_GRAYS = set(range(0, 6))
        GAMMA_EVAL_EXCLUDED_GRAYS = {0, 1, *range(248, 256)}
        
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        state = 'OFF' if profile.session_mode.startswith('VAC OFF') else 'ON'

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

            if is_on_session:
                ref_store = profile.ref_store
                # main role 기준으로 0gray 휘도 사용
                lv0_main, _, _ = store['gamma']['main']['white'].get(0, (np.nan, np.nan, np.nan))
                if np.isfinite(lv0_main):
                    self._on_lv0_current = float(lv0_main)
            
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

                    cx_ok = round(abs(d_cx), 4) <= 0.003
                    cy_ok = round(abs(d_cy), 4) <= 0.003

                    if gray in COLOR_EVAL_EXCLUDED_GRAYS:
                        self._set_item(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}")
                        self._set_item(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}")
                    else:
                        self._set_item_with_spec(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}", is_spec_ok=cx_ok)
                        self._set_item_with_spec(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}", is_spec_ok=cy_ok)

                    # ★ 여기서 ΔGamma도 간단히 추가 (VAC OFF max / 현재 ON 0gray 기준)
                    if 'd_gamma' in cols:
                        # 1) OFF 휘도 벡터 (ref_store = VAC OFF)
                        lv_off = np.zeros(256, dtype=np.float64)
                        for gg in range(256):
                            tup_off = profile.ref_store['gamma']['main']['white'].get(gg, None)
                            lv_off[gg] = float(tup_off[0]) if tup_off else np.nan

                        # 2) ON 휘도 벡터 (현재 세션 store)
                        lv_on = np.zeros(256, dtype=np.float64)
                        for gg in range(256):
                            tup_on = store['gamma']['main']['white'].get(gg, None)
                            lv_on[gg] = float(tup_on[0]) if tup_on else np.nan

                        # 3) 정규화 기준: OFF max Lv / ON 0gray Lv
                        Lv_off_max = np.nanmax(lv_off[1:])   # gray 0 제외한 max
                        Lv_on_0    = lv_on[0]

                        if (
                            np.isfinite(Lv_off_max) and
                            np.isfinite(Lv_on_0) and
                            (Lv_off_max > Lv_on_0)
                        ):
                            denom = Lv_off_max - Lv_on_0

                            # 정규화된 Y (0~1 근처로 클리핑)
                            Y_off = (lv_off - Lv_on_0) / denom
                            Y_on  = (lv_on  - Lv_on_0) / denom
                            Y_off = np.clip(Y_off, 1e-6, 1-1e-6)
                            Y_on  = np.clip(Y_on,  1e-6, 1-1e-6)

                            # gamma 계산: log(Y) / log(gray_norm)
                            gray_norm = np.linspace(0.0, 1.0, 256, dtype=np.float64)
                            gamma_off = np.full(256, np.nan, dtype=np.float64)
                            gamma_on  = np.full(256, np.nan, dtype=np.float64)

                            valid_off = (gray_norm > 0) & np.isfinite(Y_off)
                            gamma_off[valid_off] = np.log(Y_off[valid_off]) / np.log(gray_norm[valid_off])

                            valid_on = (gray_norm > 0) & np.isfinite(Y_on)
                            gamma_on[valid_on] = np.log(Y_on[valid_on]) / np.log(gray_norm[valid_on])

                            g_off = gamma_off[gray]
                            g_on  = gamma_on[gray]

                            if np.isfinite(g_off) and np.isfinite(g_on):
                                d_gamma = g_on - g_off

                                if gray in GAMMA_EVAL_EXCLUDED_GRAYS:
                                    self._set_item(table_inst1, gray, cols['d_gamma'], f"{d_gamma:.6f}")
                                else:
                                    g_ok = round(abs(d_gamma), 3) <= 0.05
                                    self._set_item_with_spec(table_inst1, gray, cols['d_gamma'], f"{d_gamma:.6f}", is_spec_ok=g_ok)

이렇게 수정했습니다. 이 메서드를 설명해주세요. 각 조건이 어떻게 분기되는지를 중심으로요
