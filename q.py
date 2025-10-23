    def _run_off_baseline_then_on(self):
        profile_off = SessionProfile(
            legend_text="VAC OFF (Ref.)",
            cie_label="data_1",
            table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
            ref_store=None
        )

        def _after_off(store_off):
            self._off_store = store_off
            logging.debug(f"VAC ON 측정 결과:\n{self._off_store}")
            self.stop_loading_animation(self.label_processing_step_1, self.movie_processing_step_1)
            self.ui.vac_label_pixmap_step_1.setPixmap(self.process_complete_pixmap)
            logging.info("[MES] VAC OFF 상태 측정 완료")
            
            logging.info("[TV CONTROL] TV VAC ON 전환")
            if not self._set_vac_active(True):
                logging.warning("[TV CONTROL] VAC ON 전환 실패 - VAC 최적화 종료")
                return
                
            # 3. DB에서 모델/주사율에 맞는 VAC Data 적용 → 읽기 → LUT 차트 갱신
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
        """
        3-a) DB에서 Panel_Maker + Frame_Rate 조합인 VAC_Data 가져오기
        3-b) TV에 쓰기 → TV에서 읽기
            → LUT 차트 갱신(reset_and_plot)
            → ON 시리즈 리셋(reset_on)
            → ON 측정 세션 시작(start_viewing_angle_session)
        """
        # 3-a) DB에서 VAC JSON 로드
        self.label_processing_step_1, self.movie_processing_step_1 = self.start_loading_animation(self.ui.vac_label_pixmap_step_2, 'processing.gif')
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
            return

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


DB에서 로드한 LUT를 tv에 바로 적용을 해서 측정을 하는게 아니라, 중간에 예측 모델로 결과값을 미리 예측 해서 보정을 하는 LUT 1차 최적화 과정을 넣으려고 해요.
보시면 감마 정확도가 낮은데 이것을 감안하여 가중치 설정을 할 수 있도록 해 주시고 우선 예측 - 보정 - 예측 - 보정 과정을 통해 LUT가 산출되면  vac_graph_rgbLUT_4, vac_table_rbgLUT_4 UI에 각각 차트와 테이블을 업데이트 하고 싶습니다.(기존엔 reading할 때 시각화함)
또 VAC OFF 측정이 끝나고 1차 최적화가 시작하는 시점에 vac_label_pixmap_step_2 라벨에 GIF를 SET 하고 싶습니다.
아래는 예측 모델입니다:

PS D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module> & C:/python310/python.exe "d:/00 업무/00 가상화기술/00 색시야각 보상 최적화/VAC algorithm/module/scripts/train_model.py"
▶ TEST with 1042 PKs

=== Train Y0: Gamma ===
⏱️ [Y0-Gamma] Linear fit: 0.1s | MSE=0.023224 R²=0.455105
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  40.7s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  41.2s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  41.4s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.7min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.7min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.8min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.5min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  31.1s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  31.2s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.5s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  31.2s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.5s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  28.3s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.5min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.2min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.2min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.2min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  53.8s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  54.1s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  54.1s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 2.0min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  58.9s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  59.8s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  59.4s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  51.6s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  51.8s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  51.5s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  59.9s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  59.6s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  59.7s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.7min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.8min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.8min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  44.7s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  44.7s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  44.0s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  27.4s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  30.3s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  30.5s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  46.1s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  47.3s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  46.3s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.9min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.9min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
⏱️ [Y0-Gamma] RF(residual) search: 11.8 min
✅ [Y0-Gamma] RF best params: {'max_depth': 17, 'max_features': 0.9722042458113105, 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 160}
✅ [Y0-Gamma] RF best R² (CV): 0.719537
🏁 [Y0-Gamma] Hybrid — MSE:0.008166 R²:0.808401
📁 saved: d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\hybrid_Gamma_model.pkl

=== Train Y0: Cx ===
⏱️ [Y0-Cx] Linear fit: 0.1s | MSE=0.000032 R²=0.304553
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  42.5s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  42.8s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  42.9s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.7min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.7min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.8min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  32.0s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  31.8s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  29.8s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  32.6s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  30.2s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  30.2s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.1min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  54.6s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  54.3s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  55.0s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  59.3s
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time= 1.0min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time= 1.0min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  53.1s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  52.7s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  53.9s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  56.5s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  57.1s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  56.9s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.7min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.7min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.1min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.1min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.8min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  45.5s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  45.1s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  45.5s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  29.1s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  29.1s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  29.0s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  45.2s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  45.6s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  45.4s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
⏱️ [Y0-Cx] RF(residual) search: 11.7 min
✅ [Y0-Cx] RF best params: {'max_depth': 17, 'max_features': 0.9722042458113105, 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 160}
✅ [Y0-Cx] RF best R² (CV): 0.953437
🏁 [Y0-Cx] Hybrid — MSE:0.000002 R²:0.963334
📁 saved: d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\hybrid_Cx_model.pkl

=== Train Y0: Cy ===
⏱️ [Y0-Cy] Linear fit: 0.1s | MSE=0.000106 R²=0.328758
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  43.7s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  44.0s
[CV] END max_depth=12, max_features=0.3248149123539492, min_samples_leaf=6, min_samples_split=4, n_estimators=207; total time=  44.6s
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.8min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.8min
[CV] END max_depth=14, max_features=0.8372343894881864, min_samples_leaf=18, min_samples_split=4, n_estimators=191; total time= 1.8min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.9min
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.9min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  31.3s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  31.4s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  29.1s
[CV] END max_depth=9, max_features=0.3454599737656805, min_samples_leaf=8, min_samples_split=2, n_estimators=177; total time=  32.0s
[CV] END max_depth=12, max_features=0.6808920093945671, min_samples_leaf=11, min_samples_split=4, n_estimators=269; total time= 1.8min
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  29.3s
[CV] END max_depth=13, max_features=0.20565304417577393, min_samples_leaf=12, min_samples_split=2, n_estimators=178; total time=  29.8s
[CV] END max_depth=12, max_features=0.9759278817295955, min_samples_leaf=15, min_samples_split=7, n_estimators=157; total time= 1.4min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=17, max_features=0.23733253057089235, min_samples_leaf=15, min_samples_split=7, n_estimators=294; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.1min
[CV] END max_depth=10, max_features=0.5059695930137302, min_samples_leaf=7, min_samples_split=2, n_estimators=250; total time= 1.1min
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  53.9s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  54.1s
[CV] END max_depth=9, max_features=0.9591084298026666, min_samples_leaf=15, min_samples_split=7, n_estimators=128; total time=  53.9s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time= 1.0min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time=  60.0s
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=12, max_features=0.6860358815211507, min_samples_leaf=8, min_samples_split=2, n_estimators=286; total time= 1.9min
[CV] END max_depth=17, max_features=0.2781376912051071, min_samples_leaf=7, min_samples_split=5, n_estimators=230; total time= 1.0min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  53.4s
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  54.0s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=11, max_features=0.3457888702304499, min_samples_leaf=7, min_samples_split=3, n_estimators=253; total time=  54.4s
[CV] END max_depth=14, max_features=0.6879973262260968, min_samples_leaf=11, min_samples_split=4, n_estimators=200; total time= 1.5min
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  55.8s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  56.0s
[CV] END max_depth=17, max_features=0.5598033066958126, min_samples_leaf=13, min_samples_split=7, n_estimators=133; total time=  56.4s
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.8min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.8min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=13, max_features=0.6373682234746237, min_samples_leaf=9, min_samples_split=6, n_estimators=265; total time= 1.7min
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  45.0s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  43.5s
[CV] END max_depth=15, max_features=0.6563551795243195, min_samples_leaf=15, min_samples_split=2, n_estimators=159; total time= 1.2min
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.0s
[CV] END max_depth=12, max_features=0.5109418317515857, min_samples_leaf=5, min_samples_split=6, n_estimators=143; total time=  45.0s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.7s
[CV] END max_depth=8, max_features=0.21250912539295516, min_samples_leaf=12, min_samples_split=2, n_estimators=255; total time=  28.5s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  46.2s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  46.8s
[CV] END max_depth=15, max_features=0.21126385817206758, min_samples_leaf=6, min_samples_split=2, n_estimators=255; total time=  46.9s
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
[CV] END max_depth=17, max_features=0.9722042458113105, min_samples_leaf=15, min_samples_split=5, n_estimators=160; total time= 1.8min
⏱️ [Y0-Cy] RF(residual) search: 11.7 min
✅ [Y0-Cy] RF best params: {'max_depth': 17, 'max_features': 0.9722042458113105, 'min_samples_leaf': 15, 'min_samples_split': 5, 'n_estimators': 160}
✅ [Y0-Cy] RF best R² (CV): 0.928867
🏁 [Y0-Cy] Hybrid — MSE:0.000006 R²:0.962731
📁 saved: d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\hybrid_Cy_model.pkl

✅ ALL DONE.

