    def _apply_vac_from_db_and_measure_on(self):
        """
        3-a) DB에서 Panel_Maker + Frame_Rate 조합인 VAC_Data 가져오기
        3-b) TV에 쓰기 → TV에서 읽기
            → LUT 차트 갱신(reset_and_plot)
            → ON 시리즈 리셋(reset_on)
            → ON 측정 세션 시작(start_viewing_angle_session)
        """
        # 3-a) DB에서 VAC JSON 로드
        self.label_processing_step_2, self.movie_processing_step_2 = self.start_loading_animation(self.ui.vac_label_pixmap_step_2, 'processing.gif')
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
            return
        
        # [ADD] 런타임 X(256×18) 생성 & 스키마 디버그 로깅
        try:
            X_runtime, lut256_norm, ctx = self._build_runtime_X_from_db_json(vac_data)
            self._debug_log_runtime_X(X_runtime, ctx, tag="[RUNTIME X from DB+UI]")
        except Exception as e:
            logging.exception("[RUNTIME X] build/debug failed")
            # 여기서 실패하면 예측/최적화 전에 스키마 문제로 조기 중단하도록 권장
            return        
        
        
        
        # ✅ 0) OFF 끝났고, 여기서 1차 예측 최적화 먼저 수행
        logging.info("[PredictOpt] 예측 기반 1차 최적화 시작")
        vac_data_to_write, _lut4096_dict = self._predictive_first_optimize(
            vac_data, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3
        )
        if vac_data_to_write is None:
            logging.warning("[PredictOpt] 실패 → 원본 DB LUT로 진행")
            vac_data_to_write = vac_data
        else:
            logging.info("[PredictOpt] 1차 최적화 LUT 생성 완료 → UI 업데이트 반영됨")

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
                self._on_store = store_on
                spec_ok = self._check_spec_pass(self._off_store, self._on_store)
                self._update_spec_views(self._off_store, self._on_store)  # ← 여기!
                if spec_ok:
                    logging.info("✅ 스펙 통과 — 종료")
                    return
                self._run_correction_iteration(iter_idx=1)

            # ── ON 측정 세션 시작 ──
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


여기서 예측-보정을 통해 생성한 new lut를 tv에 적용하고 싶습니다. 그리고, 아래 _run_correction_iteration에서 스펙 평가하는 부분을 스레드 클래스로 분리하고 싶습니다.

    def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
        logging.info(f"[CORR] iteration {iter_idx} start")

        # 1) 현재 TV LUT (캐시) 확보
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.warning("[CORR] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
            return None
        vac_dict = self._vac_dict_cache # 표준 키 dict

        # 2) 4096→256 다운샘플 (High만 수정, Low 고정)
        #    원래 키 → 표준 LUT 키로 꺼내 계산
        vac_lut = {
            "R_Low":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "R_High": np.asarray(vac_dict["RchannelHigh"], dtype=np.float32),
            "G_Low":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "G_High": np.asarray(vac_dict["GchannelHigh"], dtype=np.float32),
            "B_Low":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "B_High": np.asarray(vac_dict["BchannelHigh"], dtype=np.float32),
        }
        high_256 = {ch: self._down4096_to_256(vac_lut[ch]) for ch in ['R_High','G_High','B_High']}
        # low_256  = {ch: self._down4096_to_256(vac_lut[ch]) for ch in ['R_Low','G_Low','B_Low']}

        # 3) Δ 목표(white/main 기준): OFF vs ON 차이를 256 길이로 구성
        #    Gamma: 1..254 유효, Cx/Cy: 0..255
        d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
        # d_targets: {"Gamma":(256,), "Cx":(256,), "Cy":(256,)}

        # 4) 결합 선형계: [wG*A_Gamma; wC*A_Cx; wC*A_Cy] Δh = - [wG*ΔGamma; wC*ΔCx; wC*ΔCy]
        wG, wC = 1.0, 1.0
        A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
        b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)

        # 유효치 마스크(특히 gamma의 NaN)
        mask = np.isfinite(b_cat)
        A_use = A_cat[mask, :]
        b_use = b_cat[mask]

        # 5) 리지 해(Δh) 구하기 (3K-dim: [Rknots, Gknots, Bknots])
        #    (A^T A + λI) Δh = A^T b
        ATA = A_use.T @ A_use
        rhs = A_use.T @ b_use
        ATA[np.diag_indices_from(ATA)] += lambda_ridge
        delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

        # 6) Δcurve = Phi * Δh_channel 로 256-포인트 보정곡선 만들고 High에 적용
        K    = len(self._jac_artifacts["knots"])
        dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
        Phi  = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)
        corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

        high_256_new = {
            "R_High": (high_256["R_High"] + corr_R).astype(np.float32),
            "G_High": (high_256["G_High"] + corr_G).astype(np.float32),
            "B_High": (high_256["B_High"] + corr_B).astype(np.float32),
        }

        # 7) High 경계/단조/클램프 → 12bit 업샘플 & Low는 유지하여 "표준 dict 구성"
        for ch in high_256_new:
            self._enforce_monotone(high_256_new[ch])
            high_256_new[ch] = np.clip(high_256_new[ch], 0, 4095)

        new_lut_tvkeys = {
            "RchannelLow":  np.asarray(self._vac_dict_cache["RchannelLow"], dtype=np.float32),
            "GchannelLow":  np.asarray(self._vac_dict_cache["GchannelLow"], dtype=np.float32),
            "BchannelLow":  np.asarray(self._vac_dict_cache["BchannelLow"], dtype=np.float32),
            "RchannelHigh": self._up256_to_4096(high_256_new["R_High"]),
            "GchannelHigh": self._up256_to_4096(high_256_new["G_High"]),
            "BchannelHigh": self._up256_to_4096(high_256_new["B_High"]),
        }

        vac_write_json = self.build_vacparam_std_format(self._vac_dict_cache, new_lut_tvkeys)

        def _after_write(ok, msg):
            logging.info(f"[VAC Write] {msg}")
            if not ok:
                return
            # 쓰기 성공 → 재읽기
            logging.info(f"보정 LUT TV Reading 시작")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            if not vac_dict_after:
                logging.error("보정 후 VAC 재읽기 실패")
                return
            logging.info(f"보정 LUT TV Reading 완료")
            self.stop_loading_animation(self.label_processing_step_3, self.movie_processing_step_3)
            
            # 1) 캐시/차트 갱신
            self._vac_dict_cache = vac_dict_after
            lut_dict_plot = {k.replace("channel","_"): v
                            for k, v in vac_dict_after.items() if "channel" in k}
            # self._update_lut_chart_and_table(lut_dict_plot)
            
            # 2) ON 시리즈 리셋 (OFF는 참조 유지)
            # self.vac_optimization_gamma_chart.reset_on()
            # self.vac_optimization_cie1976_chart.reset_on()
            
            # 3) 보정 후(=ON) 측정 세션 시작
            profile_corr = SessionProfile(
                legend_text=f"CORR #{iter_idx}",   # state 판정은 'VAC OFF' prefix 여부로 하므로 여기선 ON 상태로 처리됨
                cie_label=None,                    # data_1/2 안 씀
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store          # 항상 OFF 대비 Δ를 계산
            )
            
            def _after_corr(store_corr):
                self.stop_loading_animation(self.label_processing_step_4, self.movie_processing_step_4)
                self._on_store = store_corr  # 최신 ON(보정 후) 측정 결과로 교체
                spec_ok = self._check_spec_pass(self._off_store, self._on_store)
                self._update_spec_views(self._off_store, self._on_store)
                if spec_ok:
                    logging.info("✅ 스펙 통과 — 최적화 종료")
                    return
                if iter_idx < max_iters:
                    logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
                    self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
                else:
                    logging.info("⛔ 최대 보정 횟수 도달 — 종료")

            logging.info(f"보정 LUT 기준 측정 시작")
            self.label_processing_step_4, self.movie_processing_step_4 = self.start_loading_animation(self.ui.vac_label_pixmap_step_4, 'processing.gif')
            self.start_viewing_angle_session(
                profile=profile_corr,
                gray_levels=getattr(op, "gray_levels_256", list(range(256))),
                gamma_patterns=('white',),                 # ✅ 감마는 white만 측정
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                cs_settle_ms=1000,
                on_done=_after_corr
            )

        # TV에 적용
        logging.info(f"LUT {iter_idx}차 보정 완료")
        logging.info(f"LUT {iter_idx}차 TV Writing 시작")
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)
