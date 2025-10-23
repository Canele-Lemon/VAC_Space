    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        [STEP 1] TV setting > VAC OFF → 측정(OFF baseline) + UI 업데이트

        [STEP 2] TV setting > VAC OFF → DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 writing → 측정(ON 현재) + UI 업데이트

        [STEP 3] 스펙 확인 → 통과면 종료
        
        [STEP 4] 미통과면 자코비안 기반 보정(256기준) → 4096 보간 반영 → 예측모델 검증 → OK면 → TV 적용 → 재측정 → 스펙 재확인
        [STEP 5] (필요 시 반복 2~3회만)
        """
        
        process_complete_icon_path = cf.get_normalized_path(__file__, '..', '..', 'resources/images/Icons/activered', 'radio_checked.png')
        self.process_complete_pixmap = QPixmap(process_complete_icon_path)
        self.label_processing_step_1, self.movie_processing_step_1 = self.start_loading_animation(self.ui.vac_label_pixmap_step_1, 'processing.gif')
        try:
            # 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")   # (256, 3K)
            self.A_Cx    = self._build_A_from_artifacts(artifacts, "Cx")
            self.A_Cy    = self._build_A_from_artifacts(artifacts, "Cy")
            
            # 예측 모델 로드
            self.models_Y0_bundle = self._load_prediction_models()

        except FileNotFoundError as e:
            logging.error(f"[VAC Optimization] Jacobian file not found: {e}")

        except KeyError as e:
            logging.error(f"[VAC Optimization] Missing key in artifacts: {e}")

        except Exception as e:
            logging.exception("[VAC Optimization] Unexpected error occurred")

        # 1. VAC OFF 보장 + 측정
        # 1.1 결과 저장용 버퍼 초기화 (OFF / ON 구분)
        self._off_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        self._on_store  = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        # 1.2 TV VAC OFF 하기
        logging.info("[TV CONTROL] TV VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV CONTROL] TV VAC OFF 전환 성공")    
        # 1.3 OFF 측정 세션 시작
        logging.info("[MES] VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()

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
        self.label_processing_step_2, self.movie_processing_step_2 = self.start_loading_animation(self.ui.vac_label_pixmap_step_2, 'processing.gif')
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
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

    def _predictive_first_optimize(self, vac_data_json, *, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3):
        """
        DB에서 가져온 VAC JSON을 예측모델+자코비안으로 미리 n회 보정.
        - 감마 정확도 낮음: wG(기본 0.4)로 영향 축소
        - return: (optimized_vac_json_str, lut_dict_4096)  혹은 (None, None) 실패 시
        """
        try:
            # 1) json→dict
            vac_dict = json.loads(vac_data_json)

            # 2) 4096→256
            lut256 = {
                "R_Low":  self._down4096_to_256_float(vac_dict["RchannelLow"]),
                "R_High": self._down4096_to_256_float(vac_dict["RchannelHigh"]),
                "G_Low":  self._down4096_to_256_float(vac_dict["GchannelLow"]),
                "G_High": self._down4096_to_256_float(vac_dict["GchannelHigh"]),
                "B_Low":  self._down4096_to_256_float(vac_dict["BchannelLow"]),
                "B_High": self._down4096_to_256_float(vac_dict["BchannelHigh"]),
            }

            # 3) 자코비안 준비 확인
            if not hasattr(self, "A_Gamma"):  # start에서 이미 _load_jacobian_artifacts 호출됨
                logging.error("[PredictOpt] Jacobian not prepared.")
                return None, None

            # 4) UI 메타
            panel, fr, model_year = self._get_ui_meta()

            # 5) 반복 보정
            K = len(self._jac_artifacts["knots"])
            Phi = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)

            high_R = lut256["R_High"].copy()
            high_G = lut256["G_High"].copy()
            high_B = lut256["B_High"].copy()

            for it in range(1, n_iters+1):
                # (a) 예측 ON (W)
                lut256_iter = {
                    "R_Low": lut256["R_Low"], "G_Low": lut256["G_Low"], "B_Low": lut256["B_Low"],
                    "R_High": high_R, "G_High": high_G, "B_High": high_B
                }
                y_pred = self._predict_Y0W_from_models(lut256_iter,
                            panel_text=panel, frame_rate=fr, model_year=model_year)

                # (b) Δ타깃 (예측 ON vs OFF)
                d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)

                # (c) 결합 선형계
                A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
                b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)
                mask = np.isfinite(b_cat)
                A_use = A_cat[mask,:]; b_use = b_cat[mask]

                ATA = A_use.T @ A_use
                rhs = A_use.T @ b_use
                ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
                delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

                # (d) 채널별 Δcurve = Phi @ Δh
                dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
                corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

                # (e) High 갱신 + 보정
                high_R = np.clip(self._enforce_monotone(high_R + corr_R), 0, 4095)
                high_G = np.clip(self._enforce_monotone(high_G + corr_G), 0, 4095)
                high_B = np.clip(self._enforce_monotone(high_B + corr_B), 0, 4095)

                logging.info(f"[PredictOpt] iter {it} done. (wG={wG}, wC={wC})")

            # 6) 256→4096 업샘플 (Low는 그대로, High만 갱신)
            new_lut_4096 = {
                "RchannelLow":  np.asarray(vac_dict["RchannelLow"], dtype=np.float32),
                "GchannelLow":  np.asarray(vac_dict["GchannelLow"], dtype=np.float32),
                "BchannelLow":  np.asarray(vac_dict["BchannelLow"], dtype=np.float32),
                "RchannelHigh": self._up256_to_4096(high_R),
                "GchannelHigh": self._up256_to_4096(high_G),
                "BchannelHigh": self._up256_to_4096(high_B),
            }

            # 7) UI 바로 업데이트 (차트+테이블)
            lut_plot = {
                "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
                "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
                "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
            }
            self._update_lut_chart_and_table(lut_plot)

            # 8) JSON 텍스트로 재조립 (TV write용)
            vac_json_optimized = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=new_lut_4096)

            # 9) 로딩 GIF 정지/완료 아이콘
            self.stop_loading_animation(self.label_processing_step_2, self.movie_processing_step_2)
            self.ui.vac_label_pixmap_step_2.setPixmap(self.process_complete_pixmap)

            return vac_json_optimized, new_lut_4096

        except Exception as e:
            logging.exception("[PredictOpt] failed")
            # 로딩 애니 정리
            try:
                self.stop_loading_animation(self.label_processing_step_2, self.movie_processing_step_2)
            except Exception:
                pass
            return None, None
        
    def _get_ui_meta(self):
        """
        UI 콤보값에서 패널명, 프레임레이트(Hz 제거), 모델연도(Y 제거)를 파싱해 반환.
        실패 시 0.0으로 폴백하며 로그 남김.
        """
        # Panel text
        panel_text = self.ui.vac_cmb_PanelMaker.currentText().strip()

        # Frame rate: "120Hz", "119.88 Hz" 등 → 숫자만 추출
        fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
        fr_num = 0.0
        try:
            m = re.search(r'(\d+(?:\.\d+)?)', fr_text)
            if m:
                fr_num = float(m.group(1))
            else:
                logging.warning(f"[UI META] FrameRate parsing failed: '{fr_text}' → 0.0")
        except Exception as e:
            logging.warning(f"[UI META] FrameRate parsing error for '{fr_text}': {e}")

        # Model year: "Y2024" / "2024Y" / "2024" → 숫자만 추출
        my_text = self.ui.vac_cmb_ModelYear.currentText().strip() if hasattr(self.ui, "vac_cmb_ModelYear") else ""
        model_year = 0.0
        try:
            m = re.search(r'(\d{2,4})', my_text)  # 23, 2023 등
            if m:
                model_year = float(m.group(1))
            else:
                logging.warning(f"[UI META] ModelYear parsing failed: '{my_text}' → 0.0")
        except Exception as e:
            logging.warning(f"[UI META] ModelYear parsing error for '{my_text}': {e}")

        return panel_text, fr_num, model_year
