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


여기서 예측-보정을 통해 생성한 new lut를 tv에 적용하고 싶습니다.
