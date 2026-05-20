    # ============================================================
    # 0. Initialization
    # ============================================================
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
            legend=True
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

        self.ui.vac_btn_startOptimization.clicked.connect(self.start_vac_optimization)
        self.ui.vac_btn_JSONdownload.clicked.connect(self.on_click_download_vac)
        self.ui.vac_btn_widgetexpand.clicked.connect(self.expand_VAC_Optimization_Result)

    # ============================================================
    # 1. UI Event Handlers
    # ============================================================
    def on_click_download_vac(self):
        try:
            vac_data = getattr(self, "_final_vac_data_for_download", None)

            if not vac_data:
                if getattr(self, "_vac_dict_cache", None) is None:
                    logging.error("[Download] no final VAC text and no vac cache.")
                    return
                vac_data = self._build_vacparam_std_format(
                    base_vac_dict=self._vac_dict_cache,
                    new_lut_tvkeys=None
                )

            # 파일명: VAC_YYYYmmdd_HHMMSS.json
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"VAC_{ts}.json"

            temp_dir = tempfile.gettempdir()
            path = os.path.join(temp_dir, fname)

            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(vac_data)

            logging.info(f"[Download] VAC temp file saved: {path}")

            if sys.platform.startswith("win"):
                subprocess.Popen(["notepad.exe", path], close_fds=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path], close_fds=True)
            else:
                subprocess.Popen(["xdg-open", path], close_fds=True)

        except Exception:
            logging.exception("[Download] open temp viewer failed")

    def expand_VAC_Optimization_Result(self):
        if self.expanded:
            # 현재 확장된 상태 → 축소하기
            self.ui.vac_btn_widgetexpand.setIcon(QIcon(QPixmap(":/icons/Icons/gold/chevron-left.png")))

            self.ui.vac_table_chromaticityDiff.hide()
            self.ui.vac_table_gammaLinearity.hide()
            self.ui.vac_table_colorShift_3.hide()
            current_width = self.ui.vac_widget_22.width()
            print(f"[DEBUG] Current width: {current_width}px")
            
            self.ui.vac_widget_22.setFixedWidth(300)
        
        else:
            # 현재 축소된 상태 → 확장하기
            self.ui.vac_btn_widgetexpand.setIcon(QIcon(QPixmap(":/icons/Icons/gold/chevron-right.png")))
            self.ui.vac_widget_22.setFixedWidth(600)


            self.ui.vac_table_chromaticityDiff.show()
            self.ui.vac_table_gammaLinearity.show()
            self.ui.vac_table_colorShift_3.show()

        self.expanded = not self.expanded

    # ============================================================
    # 2. VAC Optimization Workflow
    # ============================================================
    def start_vac_optimization(self):
        self._spec_policy = VACSpecPolicy()
        
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        self._fine_mode = False
        self._fine_ng_list = None
        
        self._load_jacobian_bundle_npy()
        self._load_prediction_models()
        
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")
        
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self.measure_off_ref_then_on()

    def measure_off_ref_then_on(self):
        profile_off = SessionProfile(
            session_mode="VAC OFF",
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
            self._gamma_off_vec = self.compute_gamma_series(lv_off)
            
            self._lv_off_vec = lv_off.copy()
            try:
                self._lv_off_max = float(np.nanmax(lv_off[1:]))
            except (ValueError, TypeError):
                self._lv_off_max = float('nan')
            
            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
            logging.info("[TV Control] VAC ON 전환 성공")
            
            logging.info("[Measurement] VAC ON 측정 시작")
            self.apply_predicted_vac_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off,
            on_done=_after_off
        )

    def apply_predicted_vac_and_measure_on(self):
        self._step_start(2)
        
        BASE_VAC_PK = 3025
        vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(BASE_VAC_PK)
        if base_vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        base_vac_dict = json.loads(base_vac_data)
        self._vac_dict_cache = base_vac_dict
        
        try:
            predicted_vac_data, new_lut_4096, debug_info = self._generate_predicted_vac_lut(
                base_vac_dict,
                n_iters=1,
                wG=0.4,
                wC=1.0,
                lambda_ridge=1e-3
            )
            if predicted_vac_data is None:
                raise RuntimeError("predicted_vac_data is None")
        except Exception:
            logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
            predicted_vac_data = base_vac_data
            debug_info = None
            
        predicted_vac_dict = json.loads(predicted_vac_data)
        self._vac_dict_cache = predicted_vac_dict
            
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in predicted_vac_dict.items() if "channel" in key}
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            self.send_command(self.ser_tv, 'exit')
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self.verify_vac_data_match(written_data=predicted_vac_dict, read_data=read_vac_dict)

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
                session_mode="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                pol = self._spec_policy
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self.on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=1))
                self._spec_thread.start()

            logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                on_done=_after_on
            )

        # logging.info("[VAC Writing] 예측기반 최적화 VAC 데이터 TV Writing 시작")
        # self._write_vac_to_tv(predicted_vac_data, on_finished=_after_write)

    def on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        """
        조건 1) spec_ok==True: 종료
        조건 2) (spec_ok==False) and (iter_idx < max_iters): NG Gray batch correction 반복
        """
        try:
            pol = self._spec_policy
            
            # logging
            ng_grays = []
            if metrics and "error" not in metrics:
                max_dG  = metrics.get("max_dG",  float("nan"))
                max_dCx = metrics.get("max_dCx", float("nan"))
                max_dCy = metrics.get("max_dCy", float("nan"))
                ng_grays = metrics.get("ng_grays", [])
                
                logging.info(
                    f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{pol.thr_gamma}), "
                    f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{pol.thr_c}), "
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
                
                try:
                    self._final_vac_data_for_download = self._build_vacparam_std_format(
                        base_vac_dict=self._vac_dict_cache,
                        new_lut_tvkeys=None
                    )
                except Exception:
                    logging.exception("[Download] final vac data build failed")
                    self._final_vac_data_for_download = None
                    
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            # 조건 2) (spec_ok==False) and (max_iters>0): NG Gray Correction
            self._step_fail(5)
            
            if max_iters <= 0:
                logging.info("[Evaluation] Spec NG but no further correction (max_iters≤0) - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            if iter_idx >= max_iters:
                logging.info("[Evaluation] Spec NG but 보정 횟수 초과 - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            for s in (2, 3, 4):
                self._step_set_pending(s)
                
            self.run_batch_correction_with_jacobian(
                iter_idx=iter_idx+1,
                max_iters=max_iters,
                policy=pol,
                metrics=metrics
            )
            
            # 무시하세요
            # if not getattr(self, "_failover_vac_applied3335", False):
            #         logging.info("[Failover] Spec NG — 대체 VAC(pk=3335) 적용 및 재평가 시도")
            #         self._failover_vac_applied3335 = True
            #         thr_gamma = float(thr_g) if thr_g is not None else 0.05
            #         thr_c_val = float(thr_c) if thr_c is not None else 0.003
            #         self.apply_vac_by_pk_and_re_evaluate(
            #             vac_info_pk=3336,
            #             thr_gamma=thr_gamma,
            #             thr_c=thr_c_val
            #         )
            #         return
            
        finally:
            self._spec_thread = None

    def apply_vac_by_pk_and_re_evaluate(self, vac_info_pk: int, thr_gamma: float, thr_c: float):
        """
        1) DB에서 vac_info_pk로 VAC를 가져와 TV에 씀
        2) TV에서 읽어 일치 검증
        3) ON 측정 진행
        4) SpecEvalThread로 재평가(보정 반복 없이; max_iters=0)
        """
        try:
            self._step_start(2)

            logging.info(f"[DB] VAC(pk={vac_info_pk}) 로딩 시작")
            vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(vac_info_pk)
            if base_vac_data is None:
                logging.error(f"[DB] VAC(pk={vac_info_pk}) 로딩 실패 — 대체 적용 중단")
                self._step_fail(2)
                return

            base_vac_dict = json.loads(base_vac_data)
            self._vac_dict_cache = base_vac_dict

            # (선택) LUT 차트/테이블 갱신
            try:
                lut_dict_plot = {key.replace("channel", "_"): v for key, v in base_vac_dict.items() if "channel" in key}
                self._update_lut_chart_and_table(lut_dict_plot)
            except Exception:
                logging.exception("[UI] LUT 차트/테이블 갱신 중 예외(계속 진행)")

            self._step_done(2)

            def _after_write(ok, msg):
                if not ok:
                    logging.error(f"[VAC Writing] VAC(pk={vac_info_pk}) Writing 실패: {msg}")
                    return
                logging.info(f"[VAC Writing] VAC(pk={vac_info_pk}) Writing 완료: {msg}")

                # (선택) 다운로드용 캐시
                self._last_written_base_vac_dict = base_vac_dict
                self._last_written_new_lut_tvkeys = None

                logging.info("[VAC Reading] 시작")
                self._read_vac_from_tv(_after_read)

            def _after_read(read_vac_dict):
                self.send_command(self.ser_tv, 'exit')
                if not read_vac_dict:
                    logging.error("[VAC Reading] 실패 — 재평가 중단")
                    return

                mismatch_keys = self.verify_vac_data_match(written_data=base_vac_dict, read_data=read_vac_dict)
                if mismatch_keys:
                    logging.warning(f"[VAC Reading] 데이터 불일치 — keys={mismatch_keys} — 재평가 중단")
                    return
                else:
                    logging.info("[VAC Reading] Written/Read VAC 일치")

                self._step_done(3)

                # ON 측정 & 재평가
                try:
                    self._fine_mode = False
                    self.vac_optimization_gamma_chart.reset_on()
                    self.vac_optimization_cie1976_chart.reset_on()
                except Exception:
                    logging.exception("[UI] 차트 reset 중 예외(계속 진행)")

                profile_on = SessionProfile(
                    session_mode="VAC ON",
                    cie_label="data_2",
                    table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                    ref_store=self._off_store
                )

                def _after_on(store_on):
                    logging.info("[Measurement] VAC(pk=%s) 기준 ON 측정 완료", vac_info_pk)
                    self._step_done(4)
                    self._on_store = store_on
                    self._update_last_on_lv_norm(store_on)

                    logging.info("[Evaluation] ΔCx/ΔCy/ΔGamma 재평가 시작")
                    self._step_start(5)

                    # 재평가만 수행(보정 반복 없이)
                    self._spec_thread = SpecEvalThread(
                        self._off_store, self._on_store,
                        thr_gamma=thr_gamma, thr_c=thr_c, parent=self
                    )
                    # 재평가 결과는 다시 on_spec_eval_done으로 (max_iters=0 → 보정 없음)
                    self._spec_thread.finished.connect(
                        lambda ok, met: self.on_spec_eval_done(ok, met, iter_idx=0, max_iters=0)
                    )
                    self._spec_thread.start()

                logging.info("[Measurement] VAC(pk=%s) 기준 ON 측정 시작", vac_info_pk)
                self._step_start(4)
                self.start_viewing_angle_session(profile=profile_on, on_done=_after_on)

            logging.info("[VAC Writing] VAC(pk=%s) TV Writing 시작", vac_info_pk)
            self._write_vac_to_tv(base_vac_data, on_finished=_after_write)

        except Exception:
            logging.exception(f"[Failover] VAC(pk={vac_info_pk}) 적용/재평가 중 예외 발생")

    def run_batch_correction_with_jacobian(self, iter_idx, max_iters, policy: VACSpecPolicy, lam=1e-3, metrics=None):
        logging.info(f"[Batch Correction] iteration {iter_idx} start (Jacobian dense)")

        # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
        if not hasattr(self, "_J_dense"):
            logging.error("[Batch Correction] J_dense not loaded")
            return
        
        self._load_mapping_index_gray_to_lut()
        
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.error("[Batch Correction] no VAC cache; need latest TV VAC JSON")
            return

        # 1) NG gray 리스트 / Δ 타깃 준비            
        if metrics is not None and ("ng_grays" in metrics) and ("dG" in metrics) and ("dCx" in metrics) and ("dCy" in metrics):
            ng_list = list(metrics["ng_grays"])
            d_targets = {
                "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
                "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
                "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
            }
            logging.info(f"[Batch Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
        
        else:
            dG, dCx, dCy, ng_list, *_ = SpecEvalThread.compute_gray_errors_and_ng_list(
                self._off_store, self._on_store, policy
            )
            d_targets = {
                "Gamma": dG.astype(np.float32),
                "Cx":    dCx.astype(np.float32),
                "Cy":    dCy.astype(np.float32),
            }
            logging.info(f"[Batch Correction] NG grays (recomputed by policy): {ng_list}")

        if not ng_list:
            logging.info("[Batch Correction] no NG gray → 보정 없음")
            return
    
        # 2) 현재 High LUT 확보
        vac_dict = self._vac_dict_cache
        RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
        GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
        BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

        RH = RH0.copy()
        GH = GH0.copy()
        BH = BH0.copy()

        # 3) index별 Δ 누적
        delta_acc = {"R": np.zeros_like(RH), "G": np.zeros_like(GH), "B": np.zeros_like(BH)}
        count_acc = {"R": np.zeros_like(RH, dtype=np.int32),
                    "G": np.zeros_like(GH, dtype=np.int32),
                    "B": np.zeros_like(BH, dtype=np.int32)}

        mapLUT = self._mapping_index_gray_to_lut
        
        n_gray = 256
        dR_gray = np.full(n_gray, np.nan, np.float32)
        dG_gray = np.full(n_gray, np.nan, np.float32)
        dB_gray = np.full(n_gray, np.nan, np.float32)
        corr_flag = np.zeros(n_gray, np.int32)
        wCx_gray = np.full(n_gray, np.nan, np.float32)
        wCy_gray = np.full(n_gray, np.nan, np.float32)
        wG_gray  = np.full(n_gray, np.nan, np.float32)
        
        step_gain_last = 1.0
        
        # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
        for g in ng_list:
            if 0 <= g < n_gray:
                corr_flag[g] = 1
                
            dX = self._solve_delta_rgb_for_gray(
                g,
                d_targets,
                lam=lam,
                thr_c=policy.thr_c,             # 색좌표 스펙
                thr_gamma=policy.thr_gamma,     # 감마 스펙
                base_wCx=0.5,                   # Cx 기본 가중치 (기존 0.5를 base로 사용)
                base_wCy=0.5,                   # Cy 기본 가중치
                base_wG=1.0,                    # Gamma 기본 가중치
                boost=3.0,                      # NG일 때 배율
                keep=0.2,                       # OK일 때 배율 (거의 무시)
            )
            if dX is None:
                continue

            dR, dG, dB, wCx_g, wCy_g, wG_g, step_gain = dX
            step_gain_last = step_gain
            
            dR_gray[g] = dR
            dG_gray[g] = dG
            dB_gray[g] = dB
            wCx_gray[g] = wCx_g
            wCy_gray[g] = wCy_g
            wG_gray[g]  = wG_g

            idx = int(mapLUT[g])
            if 0 <= idx < len(RH):
                delta_acc["R"][idx] += dR
                count_acc["R"][idx] += 1
            if 0 <= idx < len(GH):
                delta_acc["G"][idx] += dG
                count_acc["G"][idx] += 1
            if 0 <= idx < len(BH):
                delta_acc["B"][idx] += dB
                count_acc["B"][idx] += 1

        # 5) index별 평균 Δ 적용 + clip + monotone + 로그
        for ch, arr, arr0 in (("R", RH, RH0), ("G", GH, GH0), ("B", BH, BH0)):
            da = delta_acc[ch]
            ct = count_acc[ch]
            mask = ct > 0
            if not np.any(mask):
                logging.info(f"[Batch Correction] channel {ch}: no indices updated")
                continue
            arr[mask] = arr0[mask] + (da[mask] / ct[mask])  # 평균 Δ
            arr[:] = np.clip(arr, 0.0, 4095.0)              # clip
            self.enforce_monotone(arr)                      # 단조 증가 (i<j → LUT[i] ≤ LUT[j])

        # 6) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "RchannelHigh": RH,
            "GchannelHigh": GH,
            "BchannelHigh": BH,
        }
        for k in new_lut_4096:            
            arr = np.asarray(new_lut_4096[k], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            new_lut_4096[k] = np.clip(np.round(arr), 0, 4095).astype(np.uint16)
        
        # 7) 보정 결과 로그/저장/시각화
        df_corr = self._build_batch_corr_df(
            iter_idx=iter_idx,
            d_targets=d_targets,
            dR_gray=dR_gray, dG_gray=dG_gray, dB_gray=dB_gray,
            corr_flag=corr_flag,
            mapLUT=mapLUT,
            RH0=RH0, GH0=GH0, BH0=BH0,
            RH=RH, GH=GH, BH=BH,
            wCx_gray=wCx_gray, wCy_gray=wCy_gray, wG_gray=wG_gray,
        )
        logging.info(
            f"[Batch Correction] {iter_idx}회차 보정 결과:\n"
            + df_corr.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )
        self._save_batch_corr_df(iter_idx, df_corr, step_gain=step_gain)

        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)

        # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
        logging.info(f"[VAC Writing] LUT {iter_idx}차 보정 VAC Data TV Writing start")

        vac_corr_data = self._build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )
        vac_corr_dict = json.loads(vac_corr_data)
        self._vac_dict_cache = vac_corr_dict

        def _after_write(ok, msg):
            logging.info(f"[VAC Writing] write result: {ok} {msg}")
            if not ok:
                return
            logging.info("[VAC Reading] TV reading after write")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            self.send_command(self.ser_tv, 'exit')
            if not vac_dict_after:
                logging.error("[VAC Reading] TV read-back failed")
                return
            
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self.verify_vac_data_match(written_data=vac_corr_dict, read_data=vac_dict_after)
            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")            
            self._step_done(3)
            
            self._fine_mode = False
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_corr = SessionProfile(
                session_mode=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr
                self._update_last_on_lv_norm(store_corr)
                
                self._step_start(5)
                pol = self._spec_policy
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, policy=pol, parent=self)
                self._spec_thread.finished.connect(lambda ok, m: self.on_spec_eval_done(ok, m, iter_idx, max_iters))
                self._spec_thread.start()

            logging.info(f"[Measurement] LUT {iter_idx}차 보정 기준 re-measure start")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_corr,
                on_done=_after_corr
            )

        self._step_start(3)
        self._write_vac_to_tv(vac_corr_data, on_finished=_after_write)

    def do_gray_fix_once(self):
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
        G_on_g  = self.gamma_from_last_on_norm_at_gray(lv_on_g=lv_o, g=g)
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
            "R_Low":  self.down4096_to_256(vac_dict["RchannelLow"]),
            "G_Low":  self.down4096_to_256(vac_dict["GchannelLow"]),
            "B_Low":  self.down4096_to_256(vac_dict["BchannelLow"]),
            "R_High": self.down4096_to_256(vac_dict["RchannelHigh"]),
            "G_High": self.down4096_to_256(vac_dict["GchannelHigh"]),
            "B_High": self.down4096_to_256(vac_dict["BchannelHigh"]),
        }
        lut256_new = {k: (lut256[k] + corr[k]).astype(np.float32) for k in lut256.keys()}

        # 안전 후처리(기존 파이프라인 재사용)
        for ch in ("R","G","B"):
            Lk, Hk = f"{ch}_Low", f"{ch}_High"
            # 엔드포인트 고정
            lut256_new[Lk][0]=0.0; lut256_new[Hk][0]=0.0
            lut256_new[Lk][255]=4095.0; lut256_new[Hk][255]=4095.0
            # 역전 방지→스무딩→mid nudge→최종 안전화
            low_fixed, high_fixed = self.fix_low_high_order(lut256_new[Lk], lut256_new[Hk])
            low_s  = self.smooth_and_monotone(low_fixed, 9)
            high_s = self.smooth_and_monotone(high_fixed, 9)
            low_m, high_m = self.nudge_midpoint(low_s, high_s, max_err=3.0, strength=0.5)
            lut256_new[Lk], lut256_new[Hk] = self.finalize_channel_pair_safely(low_m, high_m)

        # ===== 6) 256→4096 ↑, JSON 구성, TV write → read → 같은 gray 재측정
        new_lut_4096 = {
            "RchannelLow":  self.up256_to_4096(lut256_new["R_Low"]),
            "GchannelLow":  self.up256_to_4096(lut256_new["G_Low"]),
            "BchannelLow":  self.up256_to_4096(lut256_new["B_Low"]),
            "RchannelHigh": self.up256_to_4096(lut256_new["R_High"]),
            "GchannelHigh": self.up256_to_4096(lut256_new["G_High"]),
            "BchannelHigh": self.up256_to_4096(lut256_new["B_High"]),
        }
        for k in new_lut_4096:
            new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

        vac_write_json = self._build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )

        def _after_write(ok, msg):
            logging.info(f"[GRAY-FIX] write: {ok} {msg}")
            if not ok:
                return self._remeasure_same_gray(g)  # 일단 재측정 시도 후 판단

            self._read_vac_from_tv(lambda vd: self._after_fix_read_and_remeasure(vd, g))

        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)

    # ============================================================
    # 3. Prediction / Jacobian Core
    # ============================================================
    def _generate_predicted_vac_lut(
        self,
        base_vac_dict: dict,
        *,
        n_iters: int = 1,
        wG: float = 0.4,          # dGamma weight
        wC: float = 1.0,          # dCx/dCy weight
        lambda_ridge: float = 1e-3,
        use_pattern_onehot: bool = False,   # 지금은 W만 쓸거면 False 권장
        patterns: tuple = ("W",),
        bypass_vac_info_pk: int = 1,
    ):
        """
        Base VAC(JSON dict)를 입력으로 받아
        1) bypass VAC(pk=1) 대비 ΔLUT(학습 feature와 동일하게) 생성
        2) ML로 dCx/dCy/dGamma per-gray 예측
        3) Jacobian(J_g: 256x3x3)으로 High LUT를 보정 (n_iters 반복)
        4) 256->4096로 업샘플 후 TV write용 JSON 생성

        Returns
        -------
        vac_json_optimized : str | None
        new_lut_4096 : dict | None
        debug_info : dict
        """

        debug_info = {
            "iters": [],
            "bypass_vac_info_pk": bypass_vac_info_pk,
        }

        try:
            # -----------------------------
            # 0) prerequisite check
            # -----------------------------
            if not hasattr(self, "_J_dense") or self._J_dense is None:
                raise RuntimeError("[PredictOpt] Jacobian bundle (_J_dense) not loaded.")

            if not hasattr(self, "models_Y0_bundle") or self.models_Y0_bundle is None:
                raise RuntimeError("[PredictOpt] Prediction models (models_Y0_bundle) not loaded.")

            # -----------------------------
            # 1) mapping index (gray->lut j)
            # -----------------------------
            self._load_mapping_index_gray_to_lut()
            idx_map = np.asarray(self._mapping_index_gray_to_lut, dtype=np.int32)  # (256,)
            if idx_map.shape[0] != 256:
                raise ValueError(f"[PredictOpt] idx_map must be (256,), got {idx_map.shape}")

            # -----------------------------
            # 2) load bypass VAC LUT (4096) from DB (pk=1)
            # -----------------------------
            vac_version_b, bypass_vac_data = self._fetch_vac_by_vac_info_pk(bypass_vac_info_pk)
            if bypass_vac_data is None:
                raise RuntimeError(f"[PredictOpt] bypass VAC fetch failed. pk={bypass_vac_info_pk}")

            bypass_vac_dict = json.loads(bypass_vac_data)

            # -----------------------------
            # 3) extract 4096 LUT arrays (base & bypass)
            # -----------------------------
            def _get_lut4096(d: dict, key: str) -> np.ndarray:
                arr = np.asarray(d[key], dtype=np.float32)
                if arr.shape[0] != 4096:
                    raise ValueError(f"[PredictOpt] {key} must be len 4096, got {arr.shape}")
                return arr

            base_RL = _get_lut4096(base_vac_dict, "RchannelLow")
            base_GL = _get_lut4096(base_vac_dict, "GchannelLow")
            base_BL = _get_lut4096(base_vac_dict, "BchannelLow")
            base_RH = _get_lut4096(base_vac_dict, "RchannelHigh")
            base_GH = _get_lut4096(base_vac_dict, "GchannelHigh")
            base_BH = _get_lut4096(base_vac_dict, "BchannelHigh")

            bp_RL = _get_lut4096(bypass_vac_dict, "RchannelLow")
            bp_GL = _get_lut4096(bypass_vac_dict, "GchannelLow")
            bp_BL = _get_lut4096(bypass_vac_dict, "BchannelLow")
            bp_RH = _get_lut4096(bypass_vac_dict, "RchannelHigh")
            bp_GH = _get_lut4096(bypass_vac_dict, "GchannelHigh")
            bp_BH = _get_lut4096(bypass_vac_dict, "BchannelHigh")

            # -----------------------------
            # 4) 256 LUT @ mapped indices
            # -----------------------------
            base_256 = {
                "R_Low":  base_RL[idx_map],
                "G_Low":  base_GL[idx_map],
                "B_Low":  base_BL[idx_map],
                "R_High": base_RH[idx_map],
                "G_High": base_GH[idx_map],
                "B_High": base_BH[idx_map],
            }
            bp_256 = {
                "R_Low":  bp_RL[idx_map],
                "G_Low":  bp_GL[idx_map],
                "B_Low":  bp_BL[idx_map],
                "R_High": bp_RH[idx_map],
                "G_High": bp_GH[idx_map],
                "B_High": bp_BH[idx_map],
            }

            # 초기 제어변수(보정 대상): High만
            high_R = base_256["R_High"].copy()
            high_G = base_256["G_High"].copy()
            high_B = base_256["B_High"].copy()

            # low는 base 그대로 (현재 설계)
            low_R = base_256["R_Low"].copy()
            low_G = base_256["G_Low"].copy()
            low_B = base_256["B_Low"].copy()

            # -----------------------------
            # 5) meta (panel onehot + fr + model_year)
            #    ※ 학습 때 VACInputBuilder meta와 동일한 방식/차원이어야 함
            # -----------------------------
            panel_text, frame_rate, model_year = self._get_ui_meta()

            # 아래 함수는 "학습 때 one-hot 차원/순서"를 그대로 맞춰줘야 합니다.
            # (PANEL_MAKER_CATEGORIES를 쓰든, 앱 내부에 동일한 매핑을 쓰든)
            panel_onehot = self.panel_text_to_onehot(panel_text).astype(np.float32)

            # pattern onehot (옵션)
            pattern_order = list(patterns)
            def _pattern_onehot(p: str) -> np.ndarray:
                v = np.zeros(len(pattern_order), dtype=np.float32)
                if p in pattern_order:
                    v[pattern_order.index(p)] = 1.0
                return v

            # -----------------------------
            # 6) helper: build X for model (per-gray)
            #    X schema == VACDataset._build_features_for_gray() 기반
            # -----------------------------
            def _build_X_y0_per_gray(d_lut_256: dict, pat: str = "W") -> np.ndarray:
                """
                d_lut_256:
                keys: R_Low,R_High,G_Low,G_High,B_Low,B_High each (256,)
                값은 "base - bypass" (raw 12bit delta @ mapped indices)
                """
                X_rows = []
                for g in range(256):
                    row = [
                        float(d_lut_256["R_Low"][g]),
                        float(d_lut_256["R_High"][g]),
                        float(d_lut_256["G_Low"][g]),
                        float(d_lut_256["G_High"][g]),
                        float(d_lut_256["B_Low"][g]),
                        float(d_lut_256["B_High"][g]),
                    ]
                    # panel maker onehot
                    row.extend(panel_onehot.tolist())
                    # fr, model_year
                    row.append(float(frame_rate))
                    row.append(float(model_year))
                    # gray_norm, LUT_j
                    row.append(float(g / 255.0))
                    row.append(float(idx_map[g]))

                    # pattern onehot (선택)
                    if use_pattern_onehot:
                        row.extend(_pattern_onehot(pat).tolist())

                    X_rows.append(row)

                return np.asarray(X_rows, dtype=np.float32)  # (256, D)

            # -----------------------------
            # 7) helper: ML predict dCx/dCy/dGamma (per-gray)
            # -----------------------------
            def _predict_y0(d_lut_256: dict, pat: str = "W"):
                X = _build_X_y0_per_gray(d_lut_256, pat=pat)

                # payload 구조: {"linear_model":..., "rf_residual":..., "target_scaler":...}
                def _hybrid_predict(model_payload: dict, X: np.ndarray) -> np.ndarray:
                    lm = model_payload["linear_model"]
                    rf = model_payload["rf_residual"]
                    ts = model_payload.get("target_scaler", {"mean": 0.0, "std": 1.0, "standardized": True})
                    y_mean = float(ts["mean"])
                    y_std  = float(ts["std"])
                    standardized = bool(ts.get("standardized", True))

                    base_s = lm.predict(X).astype(np.float32)
                    resid_s = rf.predict(X).astype(np.float32)
                    pred_s = base_s + resid_s

                    if standardized:
                        pred = pred_s * y_std + y_mean
                    else:
                        pred = pred_s
                    return pred.astype(np.float32)

                dCx_pred    = _hybrid_predict(self.models_Y0_bundle["dCx"], X)
                dCy_pred    = _hybrid_predict(self.models_Y0_bundle["dCy"], X)
                dGamma_pred = _hybrid_predict(self.models_Y0_bundle["dGamma"], X)
                return dCx_pred, dCy_pred, dGamma_pred

            # -----------------------------
            # 8) ML prediction + Jacobian correction
            # -----------------------------

            # 현재 BASE 상태에서 feature용 ΔLUT(base - bypass) 구성
            # 주의: ML 학습 기준이 BYPASS 대비 ΔLUT였으므로 여기서도 반드시 base - bypass 사용
            for it in range(1, n_iters + 1):
                d_lut_256 = {
                    "R_Low":  low_R  - bp_256["R_Low"],
                    "G_Low":  low_G  - bp_256["G_Low"],
                    "B_Low":  low_B  - bp_256["B_Low"],
                    "R_High": high_R - bp_256["R_High"],
                    "G_High": high_G - bp_256["G_High"],
                    "B_High": high_B - bp_256["B_High"],
                }

                # ML 예측: BASE VAC를 적용했을 때 BYPASS/VAC OFF 대비 예상되는 ΔCx, ΔCy, ΔGamma
                pat = pattern_order[0] if pattern_order else "W"
                dCx_pred, dCy_pred, dGamma_pred = _predict_y0(d_lut_256, pat=pat)

                # Jacobian 기반 correction
                # 목표:
                #   현재 예측 오차 ΔY_pred = [dCx, dCy, dGamma]
                #   보정 후 ΔY_new ≈ 0
                #
                # 선형 근사:
                #   ΔY_new ≈ ΔY_pred + J_g @ ΔRGB
                #
                # 따라서:
                #   J_g @ ΔRGB ≈ -ΔY_pred
                dh_R = np.zeros(256, dtype=np.float32)
                dh_G = np.zeros(256, dtype=np.float32)
                dh_B = np.zeros(256, dtype=np.float32)

                w_vec = np.array([wC, wC, wG], dtype=np.float32)

                for g in range(256):
                    Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3, 3)

                    if not np.isfinite(Jg).all():
                        continue

                    dy = np.array([
                        float(dCx_pred[g]),
                        float(dCy_pred[g]),
                        float(dGamma_pred[g]),
                    ], dtype=np.float32)

                    if not np.isfinite(dy).all():
                        continue

                    # 너무 작은 예측 오차는 보정하지 않음
                    if np.all(np.abs(dy) < 1e-8):
                        continue

                    # weighted ridge least squares
                    #
                    # minimize || W(J dRGB + dy) ||^2 + λ||dRGB||^2
                    #
                    # A dRGB = b
                    # A = J^T W^2 J + λI
                    # b = -J^T W^2 dy
                    WJ = w_vec[:, None] * Jg
                    Wy = w_vec * dy

                    A = WJ.T @ WJ + float(lambda_ridge) * np.eye(3, dtype=np.float32)
                    b = -WJ.T @ Wy

                    try:
                        dRGB = np.linalg.solve(A, b).astype(np.float32)
                    except np.linalg.LinAlgError:
                        dRGB = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

                    dh_R[g] = dRGB[0]
                    dh_G[g] = dRGB[1]
                    dh_B[g] = dRGB[2]

                # BASE High LUT에 correction을 1회만 적용
                high_R = high_R + dh_R
                high_G = high_G + dh_G
                high_B = high_B + dh_B

                # 안전 처리
                high_R = np.clip(self.enforce_monotone(high_R), 0, 4095)
                high_G = np.clip(self.enforce_monotone(high_G), 0, 4095)
                high_B = np.clip(self.enforce_monotone(high_B), 0, 4095)

                debug_info["iters"].append({
                    "iter": it,
                    "mode": "ml_prediction_then_jacobian",
                    "pred_summary": {
                        "dCx_mean": float(np.nanmean(dCx_pred)),
                        "dCy_mean": float(np.nanmean(dCy_pred)),
                        "dGamma_mean": float(np.nanmean(dGamma_pred)),
                        "dCx_abs_mean": float(np.nanmean(np.abs(dCx_pred))),
                        "dCy_abs_mean": float(np.nanmean(np.abs(dCy_pred))),
                        "dGamma_abs_mean": float(np.nanmean(np.abs(dGamma_pred))),
                        "dCx_max_abs": float(np.nanmax(np.abs(dCx_pred))),
                        "dCy_max_abs": float(np.nanmax(np.abs(dCy_pred))),
                        "dGamma_max_abs": float(np.nanmax(np.abs(dGamma_pred))),
                    },
                    "dh_summary": {
                        "dR_abs_mean": float(np.nanmean(np.abs(dh_R))),
                        "dG_abs_mean": float(np.nanmean(np.abs(dh_G))),
                        "dB_abs_mean": float(np.nanmean(np.abs(dh_B))),
                        "dR_max_abs": float(np.nanmax(np.abs(dh_R))),
                        "dG_max_abs": float(np.nanmax(np.abs(dh_G))),
                        "dB_max_abs": float(np.nanmax(np.abs(dh_B))),
                    }
                })

                logging.info(
                    f"[PredictOpt] iter {it}/{n_iters} prediction correction done. "
                    f"wG={wG}, wC={wC}, lam={lambda_ridge}"
                )

            # -----------------------------
            # 9) 256 -> 4096 upsample (High only)
            # -----------------------------            
            new_lut_tvkeys = {
                "RchannelLow":  self.to_int_list4096(base_RL),  # base_RL: (4096,)
                "GchannelLow":  self.to_int_list4096(base_GL),
                "BchannelLow":  self.to_int_list4096(base_BL),
                "RchannelHigh": self.to_int_list4096(np.round(self.up256_to_4096(high_R))),
                "GchannelHigh": self.to_int_list4096(np.round(self.up256_to_4096(high_G))),
                "BchannelHigh": self.to_int_list4096(np.round(self.up256_to_4096(high_B))),
            }

            # -----------------------------
            # 10) build json (TV write format)
            # -----------------------------
            predicted_vac_data = self._build_vacparam_std_format(
                base_vac_dict=base_vac_dict,
                new_lut_tvkeys=new_lut_tvkeys
            )

            return predicted_vac_data, new_lut_tvkeys, debug_info

        except Exception:
            logging.exception("[PredictOpt] failed")
            return None, None, debug_info

    def _solve_delta_rgb_for_gray(
        self,
        g: int,
        d_targets: dict,
        lam: float = 1e-3,
        # --- (옵션1) 기존처럼 직접 weight 지정하고 싶을 때 ---
        wCx: float | None = None,
        wCy: float | None = None,
        wG:  float | None = None,
        # --- (옵션2) NG 정도에 따라 자동 가중치 계산 ---
        thr_c: float | None = None,
        thr_gamma: float | None = None,
        base_wCx: float = 1.0,
        base_wCy: float = 1.0,
        base_wG:  float = 1.0,
        boost: float = 3.0,
        keep: float = 0.2,
    ):
        """
        주어진 gray g에서, 현재 ΔY = [dCx, dCy, dGamma]를
        자코비안 J_g를 이용해 줄이기 위한 ΔX = [ΔR_H, ΔG_H, ΔB_H]를 푼다.

        관계식:  ΔY_new ≈ ΔY + J_g · ΔX
        우리가 원하는 건 ΔY_new ≈ 0 이므로, J_g · ΔX ≈ -ΔY 를 풀어야 함.

        리지 가중 최소자승:
            argmin_ΔX || W (J_g ΔX + ΔY) ||^2 + λ ||ΔX||^2
            → (J^T W^2 J + λI) ΔX = - J^T W^2 ΔY

        - thr_c, thr_gamma가 주어지면:
            NG 여부에 따라 (base_w * boost) / (base_w * keep)로 가중치 자동 계산
        - thr_c, thr_gamma가 None 이고 wCx/wCy/wG가 주어지면:
            예전 방식처럼 고정 weight 사용
        """
        Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3,3)
        if not np.isfinite(Jg).all():
            logging.warning(f"[BATCH CORR] g={g}: J_g has NaN/inf → skip")
            return None

        dCx_g = float(d_targets["Cx"][g])
        dCy_g = float(d_targets["Cy"][g])
        dG_g  = float(d_targets["Gamma"][g])
        dy = np.array([dCx_g, dCy_g, dG_g], dtype=np.float32)  # (3,)

        # target이 NaN/Inf인 경우
        if not np.isfinite(dy).all():
            logging.warning(
                f"[BATCH CORR] g={g}: dY has NaN/inf "
                f"(dCx, dCy, dG) = ({dCx_g}, {dCy_g}, {dG_g}) → skip this gray"
            )
            return None
        
        # 이미 거의 0이면 굳이 보정 안 해도 됨
        if np.all(np.abs(dy) < 1e-6):
            return None

        # ---------------------------------------------
        # 1) 가중치 계산
        #    - 우선순위:
        #      (1) thr_c/thr_gamma가 있으면 NG 기반 자동 가중치
        #      (2) 아니면 (wCx,wCy,wG) 직접 지정값 사용
        #      (3) 둘 다 없으면 base_w* 그대로 사용
        # ---------------------------------------------
        if thr_c is not None and thr_gamma is not None:
            def w_for(err: float, thr: float, base: float) -> float:
                ratio = abs(err) / max(thr, 1e-6)
                ratio_clamped = min(ratio, 1.0)
                w = base * (keep) + (boost - keep) * ratio_clamped
                return w

            wCx_eff = w_for(dCx_g, thr_c, base_wCx)
            wCy_eff = w_for(dCy_g, thr_c, base_wCy)
            wG_eff  = w_for(dG_g,  thr_gamma, base_wG)

        elif (wCx is not None) and (wCy is not None) and (wG is not None):
            # 옛날 방식: 직접 weight 지정
            wCx_eff, wCy_eff, wG_eff = float(wCx), float(wCy), float(wG)

        else:
            # fallback: 그냥 base weight 사용
            wCx_eff, wCy_eff, wG_eff = base_wCx, base_wCy, base_wG

        w_vec = np.array([wCx_eff, wCy_eff, wG_eff], dtype=np.float32)

        # ---------------------------------------------
        # 2) 가중 least squares (기존 로직 그대로)
        # ---------------------------------------------
        WJ = w_vec[:, None] * Jg   # (3,3)
        Wy = w_vec * dy            # (3,)

        A = WJ.T @ WJ + float(lam) * np.eye(3, dtype=np.float32)  # (3,3)
        b = - WJ.T @ Wy                                           # (3,)

        try:
            dX = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

        step_gain = 1.0
        dR, dG, dB = (float(dX[0]) * step_gain,
                    float(dX[1]) * step_gain,
                    float(dX[2]) * step_gain)

        return dR, dG, dB, wCx_eff, wCy_eff, wG_eff, step_gain

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

    def _load_jacobian_bundle_npy(self):
        """
        bundle["J"]   : (256,3,3)
        bundle["n"]   : (256,)
        bundle["cond"]: (256,)
        """
        if hasattr(self, "_jac_bundle") and self._jac_bundle is not None:
            return
        
        try:
            jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_bundle_ref3008_lam0.001_20251222_142908.npy')
            if not os.path.exists(jac_path):
                raise FileNotFoundError(f"Jacobian npy not found: {jac_path}")

            bundle = np.load(jac_path, allow_pickle=True).item()
            J = np.asarray(bundle["J"], dtype=np.float32) # (256, 3, 3)
            n = np.asarray(bundle["n"], dtype=np.int32)   # (256,)
            cond = np.asarray(bundle["cond"], dtype=np.float32)

            self._jac_bundle = bundle
            self._J_dense = J
            self._J_n = n
            self._J_cond = cond

            logging.info(f"[Jacobian] dense J bundle loaded: {jac_path}, J.shape={J.shape}")

        except Exception:
            logging.exception("[Jacobian] Jacobian load failed")
            raise
        
    def _load_prediction_models(self):
        """
        hybrid_*_model.pkl 파일들을 불러와서 self.models_Y0_bundle에 저장
        (dCx / dCy / dGamma)
        """
        if hasattr(self, "models_Y0_bundle") and self.models_Y0_bundle is not None:
            return
        
        model_names = {
            "dCx": "hybrid_dCx_model.pkl",
            "dCy": "hybrid_dCy_model.pkl",
            "dGamma": "hybrid_dGamma_model.pkl",
        }

        try:
            models_dir = cf.get_normalized_path(__file__, '.', 'models')
            if not os.path.isdir(models_dir):
                raise FileNotFoundError(f"[PredictModel] 모델 디렉터리를 찾을 수 없습니다: {models_dir}")
            
            bundle = {}

            for key, fname in model_names.items():
                path = os.path.join(models_dir, fname)
                
                if not os.path.exists(path):
                    logging.error(f"[PredictModel] 모델 파일을 찾을 수 없습니다: {path}")
                    raise FileNotFoundError(f"Missing model file: {path}")
                
                try:
                    model = joblib.load(path)
                    bundle[key] = model
                    logging.info(f"[PredictModel] {key} 모델 로드 완료: {fname}")
                except Exception as e:
                    logging.exception(f"[PredictModel] {key} 모델 로드 중 오류: {e}")
                    raise

            self.models_Y0_bundle = bundle
            logging.info("[PredictModel] 모든 예측 모델 로드 완료")
            logging.debug(f"[PredictModel] keys: {list(bundle.keys())}")
        
        except Exception:
            raise
            
    def _load_prediction_model_payload(self):

        self._cx_linear  = self._cx_model["linear_model"]
        self._cx_rf      = self._cx_model["rf_residual"]
        self._cx_scaler  = self._cx_model["target_scaler"]
        self._cx_schema  = self._cx_model.get("feature_schema")

        self._cy_linear  = self._cy_model["linear_model"]
        self._cy_rf      = self._cy_model["rf_residual"]
        self._cy_scaler  = self._cy_model["target_scaler"]
        self._cy_schema  = self._cy_model.get("feature_schema")

        self._gamma_linear = self._gamma_model["linear_model"]
        self._gamma_rf     = self._gamma_model["rf_residual"]
        self._gamma_scaler = self._gamma_model["target_scaler"]
        self._gamma_schema = self._gamma_model.get("feature_schema")

        # ------------------------------------------------------------
        # payload 별 shape / 구조 로그
        # ------------------------------------------------------------
        def _log_payload_info(name, payload):
            linear = payload["linear_model"]      # Pipeline(StandardScaler + Ridge)
            rf     = payload["rf_residual"]      # RandomForestRegressor
            t_scal = payload.get("target_scaler")
            schema = payload.get("feature_schema")

            logging.info(f"[ML] ===== {name} payload info =====")

            # 1) Linear (Pipeline) 구조 & feature 수
            logging.info(f"[ML] {name}.linear_model type = {type(linear)}")
            if hasattr(linear, "steps"):
                logging.info(f"[ML] {name}.linear_model steps = {[s[0] for s in linear.steps]}")
            if hasattr(linear, "n_features_in_"):
                logging.info(f"[ML] {name}.linear_model.n_features_in_ = {linear.n_features_in_}")

            # Ridge 내부 coef shape도 참고
            try:
                ridge = linear.named_steps.get("ridge", None)
            except Exception:
                ridge = None
            if ridge is not None and hasattr(ridge, "coef_"):
                logging.info(f"[ML] {name}.ridge.coef_.shape = {np.shape(ridge.coef_)}")

            # 2) RF residual 정보
            logging.info(f"[ML] {name}.rf_residual type = {type(rf)}")
            if hasattr(rf, "n_features_in_"):
                logging.info(f"[ML] {name}.rf_residual.n_features_in_ = {rf.n_features_in_}")
            if hasattr(rf, "n_estimators"):
                logging.info(f"[ML] {name}.rf_residual.n_estimators = {rf.n_estimators}")

            # 3) 타깃 스케일러 정보 (y mean/std)
            if t_scal is not None:
                logging.info(
                    f"[ML] {name}.target_scaler "
                    f"(mean={t_scal.get('mean'):.6f}, std={t_scal.get('std'):.6f}, "
                    f"standardized={t_scal.get('standardized')})"
                )

            # 4) feature_schema (Y0 모델에만 들어 있음)
            if schema is not None:
                desc  = schema.get("desc", "")
                chs   = schema.get("channels", [])
                add_g = schema.get("add_gray_norm", False)
                add_p = schema.get("add_pattern_onehot", False)
                logging.info(f"[ML] {name}.feature_schema.desc = {desc}")
                logging.info(f"[ML] {name}.feature_schema.channels = {chs}")
                logging.info(
                    f"[ML] {name}.feature_schema.add_gray_norm={add_g}, "
                    f"add_pattern_onehot={add_p}"
                )

        _log_payload_info("Cx",    self._cx_model)
        _log_payload_info("Cy",    self._cy_model)
        _log_payload_info("Gamma", self._gamma_model)
        # ------------------------------------------------------------
        logging.info("[ML] VAC 예측 모델 payload 로딩 및 shape 로그 완료")
        
    # ============================================================
    # 4. Measurement Session
    # ============================================================
    def start_viewing_angle_session(self,
        profile: SessionProfile,
        gray_levels=op.gray_levels,
        gamma_patterns=('white',),
        colorshift_patterns=op.colorshift_patterns,
        first_gray_delay_ms=3000,
        gamma_settle_ms=1000,
        cs_settle_ms=1000,
        on_done=None,
    ):
        if gray_levels is None:
            gray_levels = op.gray_levels
        if colorshift_patterns is None:
            colorshift_patterns = op.colorshift_patterns
        if gamma_patterns is None:
            gamma_patterns=('white',)
        
        store = {
            'gamma': {
                'main': {p:{} for p in gamma_patterns}, 
                'sub': {p:{} for p in gamma_patterns}
            },
            'colorshift': {
                'main': [],
                'sub': []
            }
        }

        # 측정 작업 상태 self._sess 딕셔너리에 저장
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
        policy = self._spec_policy
        
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
            self.set_item(table_inst1, gray, cols['lv'], f"{lv_m:.6f}" if np.isfinite(lv_m) else "")
            self.set_item(table_inst1, gray, cols['cx'], f"{cx_m:.6f}" if np.isfinite(cx_m) else "")
            self.set_item(table_inst1, gray, cols['cy'], f"{cy_m:.6f}" if np.isfinite(cy_m) else "")

            # sub 테이블
            lv_s, cx_s, cy_s = store['gamma']['sub']['white'].get(gray, (np.nan, np.nan, np.nan))
            table_inst2 = self.ui.vac_table_opt_mes_results_sub
            self.set_item(table_inst2, gray, cols['lv'], f"{lv_s:.6f}" if np.isfinite(lv_s) else "")
            self.set_item(table_inst2, gray, cols['cx'], f"{cx_s:.6f}" if np.isfinite(cx_s) else "")
            self.set_item(table_inst2, gray, cols['cy'], f"{cy_s:.6f}" if np.isfinite(cy_s) else "")

            # ΔCx/ΔCy (ON 세션에서만; ref_store가 있을 때)                    
            if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
                ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
                if ref_main is not None and np.isfinite(cx_m) and np.isfinite(cy_m):
                    _, cx_r, cy_r = ref_main
                    d_cx = cx_m - cx_r
                    d_cy = cy_m - cy_r

                    if policy.should_eval_color(gray):
                        ok = policy.color_ok(d_cx, d_cy)
                        self.set_item_with_spec(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}", is_spec_ok=ok)
                        self.set_item_with_spec(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}", is_spec_ok=ok)
                    else:
                        self.set_item(table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}")
                        self.set_item(table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}")

                    # ΔGamma 실시간 계산 및 평가 (VAC OFF max / 현재 ON 0gray 기준)
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

                        if (np.isfinite(Lv_off_max) and np.isfinite(Lv_on_0) and (Lv_off_max > Lv_on_0)):
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

                                if policy.should_eval_gamma(gray):
                                    ok = policy.gamma_ok(d_gamma)
                                    self.set_item_with_spec(table_inst1, gray, cols['d_gamma'], f"{d_gamma:.6f}", is_spec_ok=ok)
                                else:
                                    self.set_item(table_inst1, gray, cols['d_gamma'], f"{d_gamma:.6f}")

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
        state = 'OFF' if profile.session_mode.startswith('VAC OFF') else 'ON'

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

        if profile.session_mode.startswith('VAC OFF'):
            # ---------- VAC OFF ----------
            # row_idx 행의
            #   col=1 → Lv(main)
            #   col=2 → u'(main)
            #   col=3 → v'(main)

            txt_lv_off = f"{lv_main:.6f}" if np.isfinite(lv_main) else ""
            txt_u_off  = f"{up_main:.6f}"  if np.isfinite(up_main)  else ""
            txt_v_off  = f"{vp_main:.6f}"  if np.isfinite(vp_main)  else ""

            self.set_item(tbl_cs_raw, row_idx, 1, txt_lv_off)
            self.set_item(tbl_cs_raw, row_idx, 2, txt_u_off)
            self.set_item(tbl_cs_raw, row_idx, 3, txt_v_off)

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

            self.set_item(tbl_cs_raw, row_idx, 4, txt_lv_on)
            self.set_item(tbl_cs_raw, row_idx, 5, txt_u_on)
            self.set_item(tbl_cs_raw, row_idx, 6, txt_v_on)

            # du'v' 계산
            # 엑셀식: =SQRT( (60deg_u' - 0deg_u')^2 + (60deg_v' - 0deg_v')^2 )
            # 여기서 main=0°, sub=60°
            duv_txt = ""
            if np.isfinite(up_main) and np.isfinite(vp_main) and np.isfinite(up_sub) and np.isfinite(vp_sub):
                dist = np.sqrt((up_sub - up_main)**2 + (vp_sub - vp_main)**2)
                duv_txt = f"{dist:.6f}"

            self.set_item(tbl_cs_raw, row_idx, 7, duv_txt)
        
    def _finalize_session(self):
        policy = self._spec_policy
        s = self._sess
        profile: SessionProfile = s['profile']
        table_main = self.ui.vac_table_opt_mes_results_main
        table_sub = self.ui.vac_table_opt_mes_results_sub
        cols = profile.table_cols

        # 1. 정면 ΔGamma 업데이트
        # 1-1) ON Gamma 계산 후 table_main의 cols['gamma'] 열에 업데이트
        lv_series_main = np.zeros(256, dtype=np.float64)
        for g in range(256):
            tup = s['store']['gamma']['main']['white'].get(g, None)
            lv_series_main[g] = float(tup[0]) if tup else np.nan
        gamma_vec = self.compute_gamma_series(lv_series_main)
        for g in range(256):
            if np.isfinite(gamma_vec[g]):
                self.set_item(table_main, g, cols['gamma'], f"{gamma_vec[g]:.6f}")

        # 1-2) (ON 세션인 경우) ΔGamma 계산 후 cols['d_gamma'] 열에 업데이트
        if profile.ref_store is not None and 'd_gamma' in cols:
            ref_lv_main = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = profile.ref_store['gamma']['main']['white'].get(g, None)
                ref_lv_main[g] = float(tup[0]) if tup else np.nan
            ref_gamma = self.compute_gamma_series(ref_lv_main)
            dG = gamma_vec - ref_gamma
            for g in range(256):
                if not np.isfinite(dG[g]):
                    continue
                if policy.should_eval_gamma(g):
                    self.set_item_with_spec(table_main, g, cols['d_gamma'], f"{dG[g]:.6f}", is_spec_ok=policy.gamma_ok(dG[g]))
                else:
                    self.set_item(table_main, g, cols['d_gamma'], f"{dG[g]:.6f}")

        # 2. 측면 slope 업데이트
        # 2-1) slope 계산을 위한 lv normalize
        lv_series_sub = np.full(256, np.nan, dtype=np.float64)
        for g in range(256):
            tup_sub = s['store']['gamma']['sub']['white'].get(g, None)
            if tup_sub:
                lv_series_sub[g] = float(tup_sub[0])

        Ynorm_sub = self.normalize_lv_series(lv_series_sub)

        # 2-2) 88-232까지 8 gray 간격 slope 계산해서 g0행에 업데이트
        is_off_session = profile.session_mode.startswith('VAC OFF')
        slope_col_idx = 3 if is_off_session else 7  # OFF의 경우 4번째 열 ON의 경우 8번째 열

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

            self.set_item(table_sub, g0, slope_col_idx, txt)

        # 표 업데이트 끝 → on_done 콜백 실행
        if callable(s['on_done']):
            try:
                s['on_done'](s['store'])
            except Exception as e:
                logging.exception(e)

    def _pause_session(self, reason:str=""):
        s = self._sess
        s['paused'] = True
        logging.info(f"[SESSION] paused. reason={reason}")

    def _resume_session(self):
        s = self._sess
        if s.get('paused', False):
            s['paused'] = False
            logging.info("[SESSION] resumed")
            QTimer.singleShot(0, lambda: self._session_step())

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

    # ============================================================
    # 5. DB / Model / File Loaders
    # ============================================================
    def _fetch_vac_by_vac_info_pk(self, pk: int):
        """
        `W_VAC_Info` 테이블에서 주어진 `PK` 값으로 `VAC_Version`과 `VAC_Data`를 가져옵니다.
        반환: (vac_version, vac_data) 또는 (None, None)
        """
        try:
            db_conn = pymysql.connect(**config.conn_params)
            cursor = db_conn.cursor()

            cursor.execute("""
                SELECT `VAC_Version`, `VAC_Data`
                FROM `W_VAC_Info`
                WHERE `PK` = %s
            """, (pk,))

            vac_row = cursor.fetchone()

            if not vac_row:
                logging.error(f"[DB] No VAC information found for PK={pk}")
                return None, None

            vac_version = vac_row[0]
            vac_data = vac_row[1]

            logging.info(f"[DB] VAC Info fetched for PK={pk} - Version: {vac_version}")
            return vac_version, vac_data

        except Exception as e:
            logging.error(f"[DB] Error while fetching VAC Info by PK={pk}: {e}")
            return None, None

    def _fetch_vac_by_model(self, panel_maker, frame_rate):
        """
        `W_VAC_Application_Status` 테이블에서 `Panel_Maker`와 `Frame_Rate` 매칭되는 `VAC_Info_PK` 조회
        → W_VAC_Info.PK=VAC_Info_PK → `VAC_Data` 읽어서 반환
        반환: (pk, vac_version, vac_data)  또는 (None, None, None)
        """
        try:
            db_conn= pymysql.connect(**config.conn_params)
            cursor = db_conn.cursor()

            cursor.execute("""
                SELECT `VAC_Info_PK`
                FROM `W_VAC_Application_Status` 
                WHERE Panel_Maker = %s AND Frame_Rate = %s
            """, (panel_maker, frame_rate))

            result = cursor.fetchone()

            if not result:
                logging.error("No VAC_Info_PK found for given Panel Maker/Frame Rate")
                return None, None, None
            
            vac_info_pk = result[0]          
            logging.debug(f"VAC_Info_PK = {vac_info_pk}")

            cursor.execute("""
                SELECT `VAC_Version`, `VAC_Data`
                FROM `W_VAC_Info`
                WHERE `PK` = %s
            """, (vac_info_pk,))

            vac_row = cursor.fetchone()

            if not vac_row:
                logging.error(f"No VAC information found for PK={vac_info_pk}")
                return None, None, None

            vac_version = vac_row[0]
            vac_data = vac_row[1]
            
            return vac_info_pk, vac_version, vac_data
        
        except Exception as e:
            logging.exception(e)
            return None, None, None
        
        finally:
            if db_conn:
                db_conn.close()

    def _load_mapping_index_gray_to_lut(self):
        if hasattr(self, "_mapping_index_gray_to_lut") and self._mapping_index_gray_to_lut is not None:
            return

        csv_path = os.path.join(os.path.dirname(__file__), "LUT_index_mapping.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Correction] LUT_index_mapping.csv not found: {csv_path}")

        df = pd.read_csv(csv_path)

        df.columns = [c.strip().lower() for c in df.columns]
        if not ("8bit gray" in df.columns and "12bit lut index" in df.columns):
            raise ValueError(f"[Correction] Unexpected columns: {df.columns}")

        gray_col = df["8bit gray"].to_numpy(dtype=np.int32)
        idx_col  = df["12bit lut index"].to_numpy(dtype=np.int32)

        # gray 순서대로 정렬 (혹시 섞여 있으면)
        sort_idx = np.argsort(gray_col)
        idx_col_sorted = idx_col[sort_idx]

        self._mapping_index_gray_to_lut = idx_col_sorted

        logging.info(f"[Correction] loaded {csv_path}, shape={df.shape}, range={idx_col_sorted.min()}~{idx_col_sorted.max()}")

    # ============================================================
    # 6. TV Controller
    # ============================================================
    def _set_vac_active(self, enable: bool) -> bool:
        try:
            logging.debug("[TV Control] 현재 VAC 적용 상태를 확인합니다.")
            current_status = self._check_vac_status()
            current_active = bool(current_status.get("activated", False))

            if current_active == enable:
                logging.info(f"[TV Control] VAC already {'ON' if enable else 'OFF'} - skipping command.")
                return True

            self.send_command(self.ser_tv, 's')
            setVACActive = (
                "luna-send -n 1 -f "
                "luna://com.webos.service.panelcontroller/setVACActive "
                f"'{{\"OnOff\":{str(enable).lower()}}}'"
            )
            self.send_command(self.ser_tv, setVACActive)
            self.send_command(self.ser_tv, 'exit', output_limit=22)
            time.sleep(3.0)
            st = self._check_vac_status()
            return bool(st.get("activated", False)) == enable
        
        except Exception as e:
            logging.error(f"[TV Control] VAC {'ON' if enable else 'OFF'} 전환 실패: {e}")
            return False

    def _check_vac_status(self):
        self.send_command(self.ser_tv, 's')
        getVACSupportstatus = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/getVACSupportStatus \'{"subscribe":true}\''
        VAC_support_status = self.send_command(self.ser_tv, getVACSupportstatus)
        VAC_support_status = self.extract_json_from_luna_send(VAC_support_status)
        self.send_command(self.ser_tv, 'exit', output_limit=22)
        
        if not VAC_support_status:
            logging.warning("[TV Control] Failed to retrieve VAC support status from TV.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        if not VAC_support_status.get("isSupport", False):
            logging.info("[TV Control] VAC is not supported on this model.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        activated = VAC_support_status.get("isActivated", False)
        logging.info(f"[TV Control] VAC 적용 상태: {activated}")
                
        return {"supported": True, "activated": activated}

    def _write_vac_to_tv(self, vac_data, on_finished):
        self._step_start(3)
        t = WriteVACdataThread(parent=self, ser_tv=self.ser_tv,
                                vacdataName=self.vacdataName, vacdata_loaded=vac_data)
        t.write_finished.connect(lambda ok, msg: on_finished(ok, msg))
        t.start()

    def _read_vac_from_tv(self, on_finished):
        t = ReadVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName)
        t.data_read.connect(lambda data: on_finished(data))
        t.error_occurred.connect(lambda err: (logging.error(err), on_finished(None)))
        t.start()
    # ============================================================
    # 7. Helper
    # ============================================================
    def _build_batch_corr_df(
        self,
        iter_idx: int,
        d_targets: dict,
        dR_gray: np.ndarray,
        dG_gray: np.ndarray,
        dB_gray: np.ndarray,
        corr_flag: np.ndarray,
        mapLUT: np.ndarray,
        RH0: np.ndarray, GH0: np.ndarray, BH0: np.ndarray,
        RH:  np.ndarray, GH:  np.ndarray, BH:  np.ndarray,
        wCx_gray: np.ndarray,
        wCy_gray: np.ndarray,
        wG_gray:  np.ndarray,
    ):
        """
        회차별 보정 결과 DF 생성 + 로그 + CSV 저장
        컬럼:
        gray | LUT idx | CORR | ΔCx | ΔCy | ΔGamma | ΔR | ΔG | ΔB |
        R_before | R_after | G_before | G_after | B_before | B_after
        """
        rows = []
        n_gray = 256

        for g in range(n_gray):
            idxLUT = int(mapLUT[g]) if 0 <= g < len(mapLUT) else -1

            row = {
                "gray": int(g),
                "LUT idx": idxLUT,
                "CORR": int(corr_flag[g]),  # 1: 이 gray는 이번 회차 보정 대상(NG), 0: OK
                "ΔCx": float(d_targets["Cx"][g]) if np.isfinite(d_targets["Cx"][g]) else np.nan,
                "ΔCy": float(d_targets["Cy"][g]) if np.isfinite(d_targets["Cy"][g]) else np.nan,
                "ΔGamma": float(d_targets["Gamma"][g]) if np.isfinite(d_targets["Gamma"][g]) else np.nan,
                "ΔR": float(dR_gray[g]) if np.isfinite(dR_gray[g]) else 0.0,
                "ΔG": float(dG_gray[g]) if np.isfinite(dG_gray[g]) else 0.0,
                "ΔB": float(dB_gray[g]) if np.isfinite(dB_gray[g]) else 0.0,
                "wCx": float(wCx_gray[g]) if np.isfinite(wCx_gray[g]) else np.nan,
                "wCy": float(wCy_gray[g]) if np.isfinite(wCy_gray[g]) else np.nan,
                "wGamma": float(wG_gray[g]) if np.isfinite(wG_gray[g]) else np.nan,
            }

            if 0 <= idxLUT < len(RH0):
                row["R_before"] = float(RH0[idxLUT])
                row["R_after"]  = float(RH[idxLUT])
                row["G_before"] = float(GH0[idxLUT])
                row["G_after"]  = float(GH[idxLUT])
                row["B_before"] = float(BH0[idxLUT])
                row["B_after"]  = float(BH[idxLUT])
            else:
                row["R_before"] = np.nan
                row["R_after"]  = np.nan
                row["G_before"] = np.nan
                row["G_after"]  = np.nan
                row["B_before"] = np.nan
                row["B_after"]  = np.nan

            rows.append(row)

        df_corr = pd.DataFrame(rows, columns=[
            "gray", "LUT idx", "CORR",
            "ΔCx", "wCx",
            "ΔCy", "wCy",
            "ΔGamma", "wGamma",
            "ΔR", "ΔG", "ΔB",
            "R_before", "R_after",
            "G_before", "G_after",
            "B_before", "B_after",
        ])

        self._last_batch_corr_df = df_corr
        
        return df_corr

    def _save_batch_corr_df(self, iter_idx: int, df_corr: pd.DataFrame, step_gain: float = None):
        artifacts_dir = os.path.join(PARENT_DIR, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # 파일 이름 예: batch_corr_iter01_gain1.0_20251110_134500.csv
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if step_gain is not None:
            filename = f"batch_corr_iter{iter_idx:02d}_gain{step_gain:.1f}_{ts}.csv"
        else:
            filename = f"batch_corr_iter{iter_idx:02d}_{ts}.csv"
        out_csv = os.path.join(artifacts_dir, filename)
        df_corr.to_csv(out_csv, index=False, encoding="utf-8-sig")
        logging.info(f"[Batch Correction] CSV saved: {out_csv}")

    def _build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict = None) -> str:
        """
        base_vac_dict : TV에서 읽은 원본 JSON(dict; 키 순서 유지 권장)
        new_lut_tvkeys: 교체할 LUT만 전달 시 병합 (TV 원 키명 그대로)
                        {"RchannelLow":[...4096], "RchannelHigh":[...], ...}
        return: TV에 바로 쓸 수 있는 탭 포맷 문자열
        """
        from collections import OrderedDict

        if not isinstance(base_vac_dict, (dict, OrderedDict)):
            raise ValueError("base_vac_dict must be dict/OrderedDict")

        od = OrderedDict(base_vac_dict)

        # 새 LUT 반영(형태/범위 보정)
        if new_lut_tvkeys:
            for k, v in new_lut_tvkeys.items():
                if k in od:
                    arr = np.asarray(v)
                    if arr.shape != (4096,):
                        raise ValueError(f"{k}: 4096 길이 필요 (현재 {arr.shape})")
                    od[k] = np.clip(arr.astype(int), 0, 4095).tolist()

        def _fmt_inline_list(lst):
            # [\t1,\t2,\t...\t]
            return "[\t" + ",\t".join(str(int(x)) for x in lst) + "\t]"

        def _fmt_list_of_lists(lst2d):
            """
            2D 리스트(예: DRV_valc_pattern_ctrl_1) 전용.
            마지막 닫힘은 ‘]\t\t]’ (쉼표 없음). 쉼표는 바깥 루프에서 1번만 붙임.
            """
            if not lst2d:
                return "[\t]"
            if not isinstance(lst2d[0], (list, tuple)):
                return _fmt_inline_list(lst2d)

            lines = []
            # 첫 행
            lines.append("[\t[\t" + ",\t".join(str(int(x)) for x in lst2d[0]) + "\t],")
            # 중간 행들
            for row in lst2d[1:-1]:
                lines.append("\t\t\t[\t" + ",\t".join(str(int(x)) for x in row) + "\t],")
            # 마지막 행(쉼표 없음) + 닫힘 괄호 정렬: “]\t\t]”
            last = "\t\t\t[\t" + ",\t".join(str(int(x)) for x in lst2d[-1]) + "\t]\t\t]"
            lines.append(last)
            return "\n".join(lines)

        def _fmt_flat_4096(lst4096):
            """
            4096 길이 LUT을 256x16으로 줄바꿈.
            마지막 줄은 ‘\t\t]’로 끝(쉼표 없음). 쉼표는 바깥에서 1번만.
            """
            a = np.asarray(lst4096, dtype=int)
            if a.size != 4096:
                raise ValueError(f"LUT 길이는 4096이어야 합니다. (현재 {a.size})")
            rows = a.reshape(256, 16)

            out = []
            # 첫 줄
            out.append("[\t" + ",\t".join(str(x) for x in rows[0]) + ",")
            # 중간 줄
            for r in rows[1:-1]:
                out.append("\t\t\t" + ",\t".join(str(x) for x in r) + ",")
            # 마지막 줄 (쉼표 X) + 닫힘
            out.append("\t\t\t" + ",\t".join(str(x) for x in rows[-1]) + "\t]")
            return "\n".join(out)

        lut_keys_4096 = {
            "RchannelLow","RchannelHigh",
            "GchannelLow","GchannelHigh",
            "BchannelLow","BchannelHigh",
        }

        keys = list(od.keys())
        lines = ["{"]

        for i, k in enumerate(keys):
            v = od[k]
            is_last_key = (i == len(keys) - 1)
            trailing = "" if is_last_key else ","

            if isinstance(v, list):
                # 4096 LUT
                if k in lut_keys_4096 and len(v) == 4096 and not (v and isinstance(v[0], (list, tuple))):
                    body = _fmt_flat_4096(v)                       # 끝에 쉼표 없음
                    lines.append(f"\"{k}\"\t:\t{body}{trailing}")  # 쉼표는 여기서 1번만
                else:
                    # 일반 1D / 2D 리스트
                    if v and isinstance(v[0], (list, tuple)):
                        body = _fmt_list_of_lists(v)               # 끝에 쉼표 없음
                        lines.append(f"\"{k}\"\t:\t{body}{trailing}")
                    else:
                        body = _fmt_inline_list(v)                 # 끝에 쉼표 없음
                        lines.append(f"\"{k}\"\t:\t{body}{trailing}")

            elif isinstance(v, (int, float)):
                if k == "DRV_valc_hpf_ctrl_1":
                    lines.append(f"\"{k}\"\t:\t\t{int(v)}{trailing}")
                else:
                    lines.append(f"\"{k}\"\t:\t{int(v)}{trailing}")

            else:
                # 혹시 모를 기타 타입
                body = json.dumps(v, ensure_ascii=False)
                lines.append(f"\"{k}\"\t:\t{body}{trailing}")

        lines.append("}")
        return "\n".join(lines)
    
    def _update_last_on_lv_norm(self, on_store):
        """
        마지막 전체 ON 측정 결과(on_store)에서
        Lv[0], max(Lv[1:]-Lv[0])를 구해 fine 보정용으로 저장.
        """
        lv_on = np.full(256, np.nan, np.float64)
        for g in range(256):
            tup = on_store['gamma']['main']['white'].get(g, None)
            if tup:
                lv_on[g] = float(tup[0])

        lv0 = lv_on[0]
        with np.errstate(invalid='ignore'):
            denom = np.nanmax(lv_on[1:] - lv0) if np.isfinite(lv0) else np.nan

        if (not np.isfinite(denom)) or denom <= 0:
            logging.warning("[FineNorm] invalid ON Lv norm (denom<=0) → fine gamma disabled")
            self._fine_lv0_on = float("nan")
            self._fine_denom_on = float("nan")
        else:
            self._fine_lv0_on = float(lv0)
            self._fine_denom_on = float(denom)
            logging.info(f"[FineNorm] updated from last ON: Lv0={lv0:.3f}, denom={denom:.3f}")

    def _extract_model_contract(self):
        """
        pkl 안에 학습시 저장해둔 메타(있다면)를 꺼내 피처 계약을 구성.
        - 기대 필드(있을 수도/없을 수도): 
        meta = {
            "panel_categories": ["HKC(H2)", "HKC(H5)", "BOE", "CSOT", "INX"],
            "pattern_order": ["W","R","G","B"],
            "feature_names": [...],            # 훈련 스크립트에서 저장했을 때
            "lut_scale": "0..1"                # LUT 정규화 기대 스케일
        }
        """
        # 기본 폴백 (훈련과 동일해야 함: 직접 학습 스크립트 확인!)
        default_panels  = ["HKC(H2)", "HKC(H5)", "BOE", "CSOT", "INX"]
        default_patterns= ["W","R","G","B"]

        # 아무 모델에서나 meta를 시도 추출
        any_model = next(iter(self.models_Y0_bundle.values()))
        meta = any_model.get("meta", {}) if isinstance(any_model, dict) else {}

        panels   = meta.get("panel_categories", default_panels)
        patterns = meta.get("pattern_order",   default_patterns)
        featnames= meta.get("feature_names",   None)
        lut_scale= meta.get("lut_scale",       "0..1")

        return {
            "panel_categories": panels,
            "pattern_order": patterns,
            "feature_names": featnames,   # 있으면 열 순서 검증에 쓰기
            "lut_scale": lut_scale
        }
    
    def _build_feature_matrix_W_checked(self, lut256_dict, *, panel_text, frame_rate, model_year):
        """
        'W' 패턴 256행 피처 매트릭스를 계약에 맞춰 생성하고,
        - n_features 일치 검사
        - 스케일/범위 검사
        - one-hot 카테고리 순서 검사
        를 수행한 뒤 (X, contract) 반환
        """
        contract = self._extract_model_contract()
        panel_cats  = contract["panel_categories"]
        pattern_ord = contract["pattern_order"]

        # 1) LUT 0..1 정규화
        def _norm01(a): 
            return np.clip(np.asarray(a, np.float32)/4095.0, 0.0, 1.0)
        R_L = _norm01(lut256_dict["R_Low"])
        R_H = _norm01(lut256_dict["R_High"])
        G_L = _norm01(lut256_dict["G_Low"])
        G_H = _norm01(lut256_dict["G_High"])
        B_L = _norm01(lut256_dict["B_Low"])
        B_H = _norm01(lut256_dict["B_High"])

        # 2) panel one-hot (훈련 순서 고정)
        panel_oh = np.zeros(len(panel_cats), np.float32)
        if panel_text in panel_cats:
            panel_oh[panel_cats.index(panel_text)] = 1.0
        else:
            logging.warning(f"[Predict/Contract] panel '{panel_text}' not in training cats {panel_cats}. (all-zero one-hot)")

        # 3) pattern one-hot 순서 확인 (W 가 index=0이어야 우리의 가정과 일치)
        if pattern_ord[0] not in ("W","White","white"):
            logging.warning(f"[Predict/Contract] training pattern order starts with {pattern_ord[0]} — expected 'W'. This must match training!")
        patt_W = np.zeros(len(pattern_ord), np.float32)
        try:
            patt_W[pattern_ord.index("W")] = 1.0
        except ValueError:
            # 훈련에서 "White"로 저장했을 수도
            if "White" in pattern_ord:
                patt_W[pattern_ord.index("White")] = 1.0
            else:
                logging.warning(f"[Predict/Contract] 'W' or 'White' not found in training pattern_order={pattern_ord}.")
                # 어쩔 수 없이 첫 칸에 1
                patt_W[0] = 1.0

        # 4) 행 단위 생성
        gray = np.arange(256, dtype=np.float32)
        gray_norm = gray/255.0
        Kp = len(panel_oh)
        Kpat = len(patt_W)

        # 기대 피처 순서: [R_L,R_H,G_L,G_H,B_L,B_H] + panel_oh + frame_rate + model_year + gray_norm + patt_W
        X = np.zeros((256, 6 + Kp + 2 + 1 + Kpat), dtype=np.float32)
        X[:,0]=R_L; X[:,1]=R_H; X[:,2]=G_L; X[:,3]=G_H; X[:,4]=B_L; X[:,5]=B_H
        X[:,6:6+Kp] = panel_oh.reshape(1,-1)
        X[:,6+Kp]   = float(frame_rate)
        X[:,6+Kp+1] = float(model_year)
        X[:,6+Kp+2] = gray_norm
        X[:,6+Kp+3:6+Kp+3+Kpat] = patt_W.reshape(1,-1)

        # 5) n_features 검증 (각 모델과 동일해야 함)
        for comp in ("Gamma","Cx","Cy"):
            lm = self.models_Y0_bundle[comp]["linear_model"]
            exp = getattr(lm, "n_features_in_", None)
            if exp is None and hasattr(lm, "coef_"):
                exp = lm.coef_.shape[1]
            if exp is not None and X.shape[1] != exp:
                logging.error(f"[Predict/Contract] n_features mismatch for {comp}: X={X.shape[1]} vs model={exp}")
            # RF도 체크
            rf = self.models_Y0_bundle[comp]["rf_residual"]
            if hasattr(rf, "n_features_in_") and rf.n_features_in_ != X.shape[1]:
                logging.error(f"[Predict/Contract] RF n_features mismatch for {comp}: X={X.shape[1]} vs RF={rf.n_features_in_}")

        # 6) 스케일/범위 로그
        def _mm(a): 
            return float(np.nanmin(a)), float(np.nanmax(a))
        logging.debug(f"[Predict/Contract] LUT(0..1) min/max — R_L{_mm(R_L)}, R_H{_mm(R_H)}, G_L{_mm(G_L)}, G_H{_mm(G_H)}, B_L{_mm(B_L)}, B_H{_mm(B_H)}")
        logging.debug(f"[Predict/Contract] meta — fr={frame_rate}, model_year={model_year}, gray_norm[0]={gray_norm[0]},[-1]={gray_norm[-1]}")
        logging.debug(f"[Predict/Contract] panel one-hot={panel_oh.tolist()}, pattern one-hot(W)={patt_W.tolist()}")
        return X, contract
    
    def _on_vac_btn_computeY_clicked(self):
        
        dR = self.ui.vac_lineEdit_dL_R.text().strip()
        dG = self.ui.vac_lineEdit_dL_G.text().strip()
        dB = self.ui.vac_lineEdit_dL_B.text().strip()
        dX = np.array([dR, dG, dB], dtype=np.float32)  # (3,)

        try:
            g = int(self.ui.vac_lineEdit_graylevel.text().strip())
        except Exception:
            g = 128
            
        if not hasattr(self, '_jac_bundle'):
            try:
                self._load_jacobian_bundle_npy()
            except Exception as e:
                logging.exception("[Jacobian] Jacobian load failed")
                return
        
        if g < 0 or g >= len(self._J_dense):
            logging.error(f"[VAC] invalid gray index g={g}, J_dense length={len(self._J_dense)}")
            return
        
        Jg = np.asarray(self._J_dense[g], dtype=np.float32)
        Jg = Jg.reshape(3, 3)
        logging.info(f"{g}에서의 Jg:\n{Jg}")

        dY = Jg @ dX
        dCx, dCy, dGamma = map(float, dY)

        self.ui.vac_lineEdit_dCx.setText(f"{dCx:+.6f}")
        self.ui.vac_lineEdit_dCy.setText(f"{dCy:+.6f}")
        self.ui.vac_lineEdit_dGamma.setText(f"{dGamma:+.6f}")

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
        if hasattr(self, "_gamma_off_vec") and hasattr(self, "_fine_lv0_on") and hasattr(self, "_fine_denom_on"):
            G_ref_g = float(self._gamma_off_vec[gray])            
            G_on_g  = self.gamma_from_last_on_norm_at_gray(lv_on_g=lv_o, g=gray)
            dG = abs(G_on_g - G_ref_g) if (np.isfinite(G_on_g) and np.isfinite(G_ref_g)) else 0.0
        else:
            dG = 0.0

        return (dCx <= thr_c) and (dCy <= thr_c) and (dG <= thr_gamma)

    def _finish_gray_fix(self, gray:int, *, pass_now: bool):
        ctx = self._sess.get('_gray_fix', None)
        if not ctx:
            self._resume_session(); return
        if pass_now or ctx['tries'] >= ctx['max']:
            logging.info(f"[GRAY-FIX] g={gray} {'PASS' if pass_now else 'MAX RETRIES'} → resume")
            self._sess['_gray_fix'] = None
            self._resume_session()
        else:
            self.do_gray_fix_once()  # 다음 재시도

    def _after_fix_read_and_remeasure(self, vac_dict_after, gray:int):
        self.send_command(self.ser_tv, 'restart panelcontroller')
        time.sleep(1.0)
        self.send_command(self.ser_tv, 'restart panelcontroller')
        time.sleep(1.0)
        self.send_command(self.ser_tv, 'exit')
        if vac_dict_after:
            self._vac_dict_cache = vac_dict_after
        self._remeasure_same_gray(gray)
    
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
        self.do_gray_fix_once()  # 첫 시도

    # ============================================================
    # 8. UI Helper
    # ============================================================
    def _set_icon_scaled(self, label, pixmap: QPixmap):
        if not label or pixmap is None or pixmap.isNull():
            return
        size = label.size()
        if size.width() <= 0 or size.height() <= 0:
            QTimer.singleShot(0, lambda: self._set_icon_scaled(label, pixmap))
            return
        scaled = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _step_label(self, step: int):
        return getattr(self.ui, f"vac_label_pixmap_step_{step}", None)

    def _ensure_step_anim_map(self):
        if not hasattr(self, "_step_anim"):
            self._step_anim = {}  # {step: (label, movie)}

    def _step_start(self, step: int):
        self._ensure_step_anim_map()
        lbl = self._step_label(step)
        if lbl is None:
            return
        if step in self._step_anim:
            return
        label_handle, movie_handle = self.start_loading_animation(lbl, 'processing.gif')
        self._step_anim[step] = (label_handle, movie_handle)

    def _step_done(self, step: int):
        self._ensure_step_anim_map()
        lbl = self._step_label(step)
        if lbl is None:
            return
        if step in self._step_anim:
            try:
                label_handle, movie_handle = self._step_anim.pop(step)
                self.stop_loading_animation(label_handle, movie_handle)
            except Exception:
                pass
        self._set_icon_scaled(lbl, self.process_complete_pixmap)

    def _step_fail(self, step: int):
        self._ensure_step_anim_map()
        lbl = self._step_label(step)
        if lbl is None:
            return
        if step in self._step_anim:
            try:
                label_handle, movie_handle = self._step_anim.pop(step)
                self.stop_loading_animation(label_handle, movie_handle)
            except Exception:
                pass
        self._set_icon_scaled(lbl, self.process_fail_pixmap)

    def _step_set_pending(self, step: int):
        lbl = self._step_label(step)
        if lbl is None:
            return
        self._set_icon_scaled(lbl, self.process_pending_pixmap)

    def _update_lut_chart_and_table(self, lut_dict):
        try:
            required = ["R_Low", "R_High", "G_Low", "G_High", "B_Low", "B_High"]
            for k in required:
                if k not in lut_dict:
                    logging.error(f"missing key: {k}")
                    return
                if len(lut_dict[k]) != 4096:
                    logging.error(f"invalid length for {k}: {len(lut_dict[k])} (expected 4096)")
                    return
                
            df = pd.DataFrame({
                "R_Low":  lut_dict["R_Low"],
                "R_High": lut_dict["R_High"],
                "G_Low":  lut_dict["G_Low"],
                "G_High": lut_dict["G_High"],
                "B_Low":  lut_dict["B_Low"],
                "B_High": lut_dict["B_High"],
            })
            self.update_rgbchannel_table(df, self.ui.vac_table_rbgLUT_4)
            self.vac_optimization_lut_chart.reset_and_plot(lut_dict)
        
        except Exception as e:
            logging.exception(e)

    def _get_ui_meta(self):
        """
        UI 콤보에서 panel / frame_rate / model_year(두 자리 float)를 가져온다.
        - FrameRate: "60Hz", "119.88 Hz" 등에서 숫자만 추출
        - ModelYear: 기본 형식 "Y26" ← 권장
        (안전장치로 "26Y"도 허용하되, 우선순위는 "Y{2자리}" 매칭)
        """
        import re
        panel_text = ""
        fr_val = 0.0
        my_val = 0.0

        # Panel
        try:
            panel_text = self.ui.vac_cmb_PanelMaker.currentText().strip()
        except Exception as e:
            logging.debug(f"[UI META] Panel text 읽기 실패: {e}")

        # FrameRate
        try:
            fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
            m = re.search(r'(\d+(?:\.\d+)?)', fr_text)  # 숫자만
            fr_val = float(m.group(1)) if m else 0.0
        except Exception as e:
            logging.debug(f"[UI META] FrameRate 파싱 에러: {e}")

        # ModelYear (우선: 'Y26' 정확 매칭 → 폴백: '26Y')
        try:
            if hasattr(self, "ui") and hasattr(self.ui, "vac_cmb_ModelYear"):
                my_text = self.ui.vac_cmb_ModelYear.currentText().strip()
                # 우선순위 1: 'Y' + 2자리
                m1 = re.match(r'^[Yy]\s*(\d{2})$', my_text)
                if m1:
                    my_val = float(m1.group(1))
                else:
                    # 우선순위 2: 2자리 + 'Y'
                    m2 = re.match(r'^(\d{2})\s*[Yy]$', my_text)
                    if m2:
                        my_val = float(m2.group(1))
                    else:
                        logging.debug(f"[UI META] ModelYear 형식 비정상: '{my_text}' → 0.0")
            else:
                logging.debug("[UI META] ModelYear 콤보 없음 → 0.0")
        except Exception as e:
            logging.debug(f"[UI META] ModelYear 파싱 에러: {e}")

        logging.debug(f"[UI META] panel_maker='{panel_text}', frame_rate='{fr_val}Hz', model_year='Y{int(my_val):02d}'")
        
        return panel_text, fr_val, my_val

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

        # ===== 공통: white 시리즈 추출 (정면 / 측면) =====
        def _extract_white(series_store, view_angle="front"):
            """
            Extract Lv, Cx, Cy arrays from white pattern data.
            view_angle: "front" → use 'main' data, "side" → use 'sub' data
            """
            lv = np.full(256, np.nan, np.float64)
            cx = np.full(256, np.nan, np.float64)
            cy = np.full(256, np.nan, np.float64)

            key = "main" if view_angle == "front" else "sub"

            for g in range(256):
                tup = series_store['gamma'][key]['white'].get(g, None)
                if tup:
                    lv[g] = float(tup[0])
                    cx[g] = float(tup[1])
                    cy[g] = float(tup[2])

            return lv, cx, cy

        # 정면 기준 (chromaticity / gamma spec)
        lv_off, cx_off, cy_off = _extract_white(off_store, view_angle="front")
        lv_on,  cx_on,  cy_on  = _extract_white(on_store,  view_angle="front")

        # 측면 기준 (gamma linearity 용)
        lv_off_side, cx_off_side, cy_off_side = _extract_white(off_store, view_angle="side")
        lv_on_side,  cx_on_side,  cy_on_side  = _extract_white(on_store,  view_angle="side")

        # ===== 1) ChromaticityDiff 표: pass/total =====
        G_off = self.compute_gamma_series(lv_off)
        G_on  = self.compute_gamma_series(lv_on)

        dG  = G_on - G_off
        dCx = cx_on - cx_off
        dCy = cy_on - cy_off

        # --- ΔCx/ΔCy: 소수점 4번째 자리 반올림 기준, edge gray(0,1,254,255) 완전 제외 ---
        def _pass_total_chroma(d_arr, thr):
            mask = np.isfinite(d_arr)
            # edge grays 제외
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

        # --- ΔGamma: 소수점 3번째 자리 반올림 기준, edge gray(0,1,254,255) 완전 제외 ---
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
        ok_g,  tot_g  = _pass_total_gamma(dG,  thr_gamma)

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

        logging.debug(
            f"{iter_idx}차 보정 결과: "
            f"Cx:{ok_cx}/{tot_cx}, "
            f"Cy:{ok_cy}/{tot_cy}, "
            f"Gamma:{ok_g}/{tot_g}"
        )

        # ===== 2) ChromaticityDiff 차트: Cx/Cy vs gray (OFF/ON, 정면 기준) =====
        x = np.arange(256)

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

        # y축 autoscale with margin 1.1 (chromaticity)
        all_y = np.concatenate([
            np.asarray(cx_off, dtype=np.float64),
            np.asarray(cx_on,  dtype=np.float64),
            np.asarray(cy_off, dtype=np.float64),
            np.asarray(cy_on,  dtype=np.float64),
        ])
        all_y = all_y[np.isfinite(all_y)]
        if all_y.size > 0:
            ymin = float(np.min(all_y))
            ymax = float(np.max(all_y))
            center = 0.5 * (ymin + ymax)
            half = 0.5 * (ymax - ymin)
            if half <= 0:
                half = max(0.001, abs(center) * 0.05)
            half *= 1.1  # 10% margin
            new_min = center - half
            new_max = center + half

            ax_chr = self.vac_optimization_chromaticity_chart.ax
            cs.MatFormat_Axis(
                ax_chr,
                min_val=np.float64(new_min),
                max_val=np.float64(new_max),
                tick_interval=None,
                axis='y'
            )
            ax_chr.relim()
            ax_chr.autoscale_view(scalex=False, scaley=False)
            self.vac_optimization_chromaticity_chart.canvas.draw()

        # ===== 3) GammaLinearity 표: 88~232, 8gray 블록 평균 슬로프 (측면 기준) =====
        def _normalized_luminance(lv_vec):
            """
            lv_vec: (256,) 절대 휘도 [cd/m2]
            return: (256,) 0~1 정규화된 휘도
                    Ynorm[g] = (Lv[g] - Lv[0]) / (max(Lv[1:]-Lv[0]))
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
                    g0 = block start, g1 = g0+step
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

                d_gray_norm = (g1 - g0) / 255.0

                if np.isfinite(y0) and np.isfinite(y1) and d_gray_norm > 0:
                    slope = abs(y1 - y0) / d_gray_norm
                else:
                    slope = np.nan

                mids.append(g0 + (g1 - g0) / 2.0)  # 예: 88~96 -> 92.0
                slopes.append(slope)

            return np.asarray(mids, dtype=np.float64), np.asarray(slopes, dtype=np.float64)

        mids_off, slopes_off = _block_slopes(lv_off_side, g_start=88, g_stop=232, step=8)
        mids_on,  slopes_on  = _block_slopes(lv_on_side,  g_start=88, g_stop=232, step=8)

        avg_off = float(np.nanmean(slopes_off)) if np.isfinite(slopes_off).any() else float('nan')
        avg_on  = float(np.nanmean(slopes_on )) if np.isfinite(slopes_on ).any() else float('nan')

        tbl_gl = self.ui.vac_table_gammaLinearity
        _set_text(tbl_gl, 1, 1, f"{avg_off:.2f}")  # ★ 소수점 둘째 자리까지
        _set_text(tbl_gl, 1, 2, f"{avg_on:.2f}")   # ★ 소수점 둘째 자리까지

        # ===== 4) GammaLinearity 차트: 블록 중심 x (= g+4), dot+line =====
        # 라인 세팅 (자동 스케일링은 직접 처리할 거라 autoscale=False)
        self.vac_optimization_gammalinearity_chart.set_series(
            "OFF_slope8",
            mids_off,
            slopes_off,
            marker='o',
            linestyle='-',
            label='OFF slope(8)',
            autoscale=False
        )
        off_ln = self.vac_optimization_gammalinearity_chart.lines["OFF_slope8"]
        off_ln.set_color('black')
        off_ln.set_markersize(3)

        self.vac_optimization_gammalinearity_chart.set_series(
            "ON_slope8",
            mids_on,
            slopes_on,
            marker='o',
            linestyle='-',
            label='ON slope(8)',
            autoscale=False
        )
        on_ln = self.vac_optimization_gammalinearity_chart.lines["ON_slope8"]
        on_ln.set_color('red')
        on_ln.set_markersize(3)

        # y축 autoscale with margin 1.1 + tick 5개
        all_slopes = np.concatenate([
            np.asarray(slopes_off, dtype=np.float64),
            np.asarray(slopes_on,  dtype=np.float64),
        ])
        all_slopes = all_slopes[np.isfinite(all_slopes)]
        if all_slopes.size > 0:
            ymin = float(np.min(all_slopes))
            ymax = float(np.max(all_slopes))
            center = 0.5 * (ymin + ymax)
            half = 0.5 * (ymax - ymin)
            if half <= 0:
                half = max(0.001, abs(center) * 0.05)
            half *= 1.1  # 10% margin
            new_min = center - half
            new_max = center + half

            # ★ XYChart 메서드 사용: y축 범위 + tick 5개
            self.vac_optimization_gammalinearity_chart.set_y_axis_range(
                new_min,
                new_max,
                tick_count=5
            )

        # ===== 5) ColorShift(4종) 표 & 6) 묶음 막대 =====
        want_names = ['Dark Skin', 'Light Skin', 'Asian', 'Western']
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

                lv0, u0, v0 = state_store['colorshift']['main'][idx]
                lv6, u6, v6 = state_store['colorshift']['sub'][idx]

                if not all(np.isfinite([u0, v0, u6, v6])):
                    arr.append(np.nan)
                    continue

                d = float(np.sqrt((u6 - u0)**2 + (v6 - v0)**2))
                arr.append(d)

            return np.array(arr, dtype=np.float64)  # [DarkSkin, LightSkin, Asian, Western]

        duv_off = _delta_uv_for_state(off_store)
        duv_on  = _delta_uv_for_state(on_store)

        mean_off = float(np.nanmean(duv_off)) if np.isfinite(duv_off).any() else float('nan')
        mean_on  = float(np.nanmean(duv_on )) if np.isfinite(duv_on ).any() else float('nan')

        tbl_cs = self.ui.vac_table_colorShift_3
        # OFF (소수점 3째 자리까지)
        _set_text(tbl_cs, 1, 1, f"{duv_off[0]:.3f}")   # DarkSkin
        _set_text(tbl_cs, 2, 1, f"{duv_off[1]:.3f}")   # LightSkin
        _set_text(tbl_cs, 3, 1, f"{duv_off[2]:.3f}")   # Asian
        _set_text(tbl_cs, 4, 1, f"{duv_off[3]:.3f}")   # Western
        _set_text(tbl_cs, 5, 1, f"{mean_off:.3f}")     # 평균

        # ON (소수점 3째 자리까지)
        _set_text(tbl_cs, 1, 2, f"{duv_on[0]:.3f}")
        _set_text(tbl_cs, 2, 2, f"{duv_on[1]:.3f}")
        _set_text(tbl_cs, 3, 2, f"{duv_on[2]:.3f}")
        _set_text(tbl_cs, 4, 2, f"{duv_on[3]:.3f}")
        _set_text(tbl_cs, 5, 2, f"{mean_on:.3f}")

        # 묶음 막대 차트 갱신
        self.vac_optimization_colorshift_chart.update_grouped(
            data_off=list(np.nan_to_num(duv_off, nan=0.0)),
            data_on =list(np.nan_to_num(duv_on,  nan=0.0))
        )

    def _ensure_row_count(self, table, row_idx):
        if table.rowCount() <= row_idx:
            old_rows = table.rowCount()
            table.setRowCount(row_idx + 1)

            # 새로 열린 구간에 대해서 header label 채우기
            vh = table.verticalHeader()
            for r in range(old_rows, row_idx + 1):
                vh_item = vh.model().headerData(r, Qt.Vertical)
                # headerData가 비어있을 때만 세팅 (중복세팅 방지)
                if vh_item is None or str(vh_item) == "":
                    vh.setSectionResizeMode(r, QHeaderView.Fixed)  # optional: 높이 고정 유지
                    table.setVerticalHeaderItem(r, QTableWidgetItem(str(r)))


    # ============================================================
    # 9. Utility-like Methods
    # ============================================================
    def enforce_monotone(self, arr):
        # 제자리 누적 최대치
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                arr[i] = arr[i-1]
        return arr
    
    def to_int_list4096(self, x) -> list:
        a = np.asarray(x, dtype=np.float32)

        if a.size != 4096:
            raise ValueError(f"4096 LUT size mismatch: {a.size}")

        a = np.nan_to_num(a, nan=0.0, posinf=4095.0, neginf=0.0)
        a = np.clip(np.round(a), 0, 4095).astype(np.uint16)

        return a.tolist()
    
    def normalize_lv_series(self, lv_vec_256, *, eps=0.0):
        """
        Lv(256) -> normalized Y(256)

        Y = (Lv - Lv0) / max(Lv[1:] - Lv0)

        - Lv0: gray 0 휘도
        - denom: (Lv[1:] - Lv0)의 최대값
        - denom이 0/NaN이면 전부 NaN 반환
        - eps>0이면 Y를 [eps, 1-eps]로 클리핑 (gamma 계산 안정화용)
        """
        lv = np.asarray(lv_vec_256, dtype=np.float64)
        y = np.full(256, np.nan, dtype=np.float64)

        if lv.size < 256:
            # 방어(필요하면 제거 가능)
            tmp = np.full(256, np.nan, dtype=np.float64)
            tmp[:lv.size] = lv
            lv = tmp

        lv0 = lv[0]
        denom = np.nanmax(lv[1:] - lv0)
        if not np.isfinite(denom) or denom <= 0:
            return y

        y = (lv - lv0) / denom

        if eps and eps > 0:
            y = np.clip(y, eps, 1.0 - eps)

        return y
    
    def compute_gamma_series(self, lv_vec_256):
        lv = np.asarray(lv_vec_256, dtype=np.float64)
        gamma = np.full(256, np.nan, dtype=np.float64)
        
        nor = self.normalize_lv_series(lv)

        gray = np.arange(256, dtype=np.float64)
        gray_norm = gray / 255.0
        
        valid = (gray >= 1) & (gray <= 254) & (nor > 0) & np.isfinite(nor)

        with np.errstate(divide='ignore', invalid='ignore'):
            gamma[valid] = np.log(nor[valid]) / np.log(gray_norm[valid])
        
        return gamma

    def interp_256_to_4096_with_map(self, ctrl_vals_256: np.ndarray, idx_map: np.ndarray) -> np.ndarray:
        """
        256개 control 값(ctrl_vals_256)을
        12bit LUT index 매핑(idx_map, shape=(256,))을 기준으로
        4096포인트로 선형 보간.

        - idx_map[k] 위치에서 LUT 값은 ctrl_vals_256[k]
        - idx_map[k] ~ idx_map[k+1] 구간은 선형 interpolation
        - 첫 구간 이전 / 마지막 구간 이후는 양 끝값을 유지
        """
        ctrl = np.asarray(ctrl_vals_256, dtype=np.float32)
        idx  = np.asarray(idx_map,       dtype=np.int32)

        if ctrl.shape[0] != 256 or idx.shape[0] != 256:
            raise ValueError(f"interp_256_to_4096_with_map: ctrl_len={ctrl.shape[0]}, idx_len={idx.shape[0]} (expected 256)")

        n_lut = 4096
        lut = np.empty(n_lut, dtype=np.float32)

        # 1) 앞/뒤 구간은 양 끝값으로 채움
        first_i = int(idx[0])
        last_i  = int(idx[-1])

        # 앞쪽
        if first_i > 0:
            lut[:first_i] = float(ctrl[0])

        # 뒤쪽
        if last_i < n_lut:
            lut[last_i:] = float(ctrl[-1])

        # 2) 각 구간 [idx[k], idx[k+1]] 선형 보간
        for k in range(255):
            i0 = int(idx[k])
            i1 = int(idx[k + 1])
            v0 = float(ctrl[k])
            v1 = float(ctrl[k + 1])

            if i1 < i0:   # 방어 코드 (정렬 안 되어 있을 경우)
                i0, i1 = i1, i0
                v0, v1 = v1, v0

            if i0 == i1:
                # 같은 인덱스면 그냥 값만 넣고 계속
                lut[i0] = v0
                continue

            length = i1 - i0
            # i0 ~ i1 (inclusive) 구간 선형 보간
            t = np.linspace(0.0, 1.0, length + 1, dtype=np.float32)
            lut[i0:i1 + 1] = v0 + (v1 - v0) * t

        return lut

    def verify_vac_data_match(self, written_data: dict, read_data: dict) -> list:
        """
        Compare written VAC data with read VAC data from TV.
        Returns: a list of keys that do not match
        """
        mismatch_keys = []

        for key in written_data:
            if key not in read_data:
                mismatch_keys.append(key)
            elif written_data[key] != read_data[key]:
                mismatch_keys.append(key)

        if mismatch_keys:
            logging.warning(f"[VAC Reading] Mismatched keys found: {mismatch_keys}")
        else:
            logging.info("[VAC Reading] VAC data successfully verified - no mismatches.")

        return mismatch_keys

    def panel_text_to_onehot(self, panel_text: str) -> np.ndarray:
        PANEL_MAKER_CATEGORIES = [['HKC(H2)', 'HKC(H5)', 'BOE', 'CSOT', 'INX']]
        cats = PANEL_MAKER_CATEGORIES[0]

        v = np.zeros(len(cats), dtype=np.float32)

        if panel_text is None:
            return v

        p = str(panel_text).strip()
        if p in cats:
            v[cats.index(p)] = 1.0

        return v

    def gamma_from_last_on_norm_at_gray(self, lv_on_g: float, g: int) -> float:
        """
        마지막 전체 ON 측정의 Lv0/denom (self._fine_lv0_on / _fine_denom_on)을
        정규화 기준으로 사용하여, 현재 Lv_on(g)에서 γ를 추정.
        """
        lv0  = getattr(self, "_fine_lv0_on", float("nan"))
        denom = getattr(self, "_fine_denom_on", float("nan"))

        if (not np.isfinite(lv0)) or (not np.isfinite(denom)) or denom <= 0:
            return float("nan")

        nor = (lv_on_g - lv0) / denom
        gray_norm = g / 255.0

        if (not np.isfinite(nor)) or nor <= 0:
            return float("nan")
        if gray_norm <= 0 or gray_norm >= 1:
            return float("nan")

        return float(np.log(nor) / np.log(gray_norm))

    def smooth_and_monotone(self, arr, win=9):
        """
        고주파(지글지글) 제거:
        1) 이동평균으로 부드럽게
        2) 단조 증가 강제 (enforce_monotone보다 먼저 완만화해서 계단 줄이기)
        arr: np.array(float32, len=256, 0~4095 스케일)
        """
        arr = np.asarray(arr, dtype=np.float32)
        half = win // 2
        tmp = np.empty_like(arr)
        n = len(arr)
        for i in range(n):
            i0 = max(0, i-half)
            i1 = min(n, i+half+1)
            tmp[i] = np.mean(arr[i0:i1])
        # 단조 정리 (non-decreasing)
        for i in range(1, n):
            if tmp[i] < tmp[i-1]:
                tmp[i] = tmp[i-1]
        return tmp

    def fix_low_high_order(self, low_arr, high_arr):
        """
        각 gray마다 Low > High이면 둘 다 중간값(mid)로 맞춰서 역전 없애기.
        반환 (low_fixed, high_fixed)
        """
        low  = np.asarray(low_arr , dtype=np.float32).copy()
        high = np.asarray(high_arr, dtype=np.float32).copy()
        for g in range(len(low)):
            if low[g] > high[g]:
                mid = 0.5 * (low[g] + high[g])
                low[g]  = mid
                high[g] = mid
        return low, high

    def nudge_midpoint(self, low_arr, high_arr, max_err=3.0, strength=0.5):
        """
        (Low+High)/2 평균 밝기가 gray(이상적으로 y=x VAC OFF)에서 너무 벗어난 곳만
        살짝 당겨서 감마 튐 억제.
        - max_err: 허용 오차(12bit count). 그 이상만 수정
        - strength: 보정 강도 (0.5면 에러의 절반만 교정)
        반환 (low_adj, high_adj)
        """
        low  = np.asarray(low_arr , dtype=np.float32).copy()
        high = np.asarray(high_arr, dtype=np.float32).copy()

        gray_12 = (np.arange(256, dtype=np.float32) * 4095.0) / 255.0
        avg     = 0.5 * (low + high)
        err     = avg - gray_12  # 양수면 평균이 너무 밝음

        mask = np.abs(err) > max_err
        adj  = err * strength   # 양수면 아래로 당김

        high[mask] -= adj[mask]
        low [mask] -= adj[mask]

        return low, high
    
    def finalize_channel_pair_safely(self, low_arr, high_arr):
        """
        마지막 안전화 단계:
        1) 다시 Low>High 방지
        2) 단조 증가 강제 (enforce_monotone)
        3) 0/255 엔드포인트 강제: 0→0, 255→4095
        4) 0~4095 clip
        """
        low  = np.asarray(low_arr , dtype=np.float32).copy()
        high = np.asarray(high_arr, dtype=np.float32).copy()

        # (1) 다시 Low>High 방지
        for g in range(len(low)):
            if low[g] > high[g]:
                mid = 0.5 * (low[g] + high[g])
                low[g]  = mid
                high[g] = mid

        # (2) 단조 증가
        low  = self.enforce_monotone(low)
        high = self.enforce_monotone(high)

        # (3) 엔드포인트 고정
        low[0]  = 0.0
        high[0] = 0.0
        low[-1]  = 4095.0
        high[-1] = 4095.0

        # (4) clip
        low  = np.clip(low ,  0.0, 4095.0)
        high = np.clip(high, 0.0, 4095.0)

        return low.astype(np.float32), high.astype(np.float32)

    def down4096_to_256(self, arr4096):
        arr4096 = np.asarray(arr4096, dtype=np.float32)
        idx = np.round(np.linspace(0, 4095, 256)).astype(int)
        return arr4096[idx]

    def up256_to_4096(self, arr256):
        arr256 = np.asarray(arr256, dtype=np.float32)
        x_small = np.linspace(0, 1, 256)
        x_big   = np.linspace(0, 1, 4096)
        return np.interp(x_big, x_small, arr256).astype(np.float32)
    
    def set_item(self, table, row, col, value):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))

        table.scrollToItem(item, QAbstractItemView.PositionAtTop)
        
    def set_item_with_spec(self, table, row, col, value, *, is_spec_ok: bool):
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
