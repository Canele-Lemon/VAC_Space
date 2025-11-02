    #################################################################################################
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
    
    def _load_jacobian_artifacts(self):
        """
        jacobian_().pkl 파일을 불러와서 artifacts 딕셔너리로 반환
        """
        jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_INX_60_K33.pkl')
        if not os.path.exists(jac_path):
            logging.error(f"[Jacobian] 파일을 찾을 수 없습니다: {jac_path}")
            raise FileNotFoundError(f"Jacobian model not found: {jac_path}")

        artifacts = joblib.load(jac_path)
        logging.info(f"[Jacobian] 모델 로드 완료: {jac_path}")
        print("======================= artifacts 구조 확인 =======================")
        logging.debug(f"Artifacts keys: {artifacts.keys()}")
        logging.debug(f"Components: {artifacts['components'].keys()}")
        return artifacts
    
    def _build_A_from_artifacts(self, artifacts, comp: str):
        """
        저장된 자코비안 pkl로부터 A 행렬 (ΔY ≈ A·Δh) 복원
        이제 Δh = [ΔR_Low_knots, ΔG_Low_knots, ΔB_Low_knots,
                ΔR_High_knots,ΔG_High_knots,ΔB_High_knots] (총 6*K)
        반환 A shape: (256, 6*K)
        """
        knots = np.asarray(artifacts["knots"], dtype=np.int32)
        comp_obj = artifacts["components"][comp]

        coef  = np.asarray(comp_obj["coef"], dtype=np.float32)
        scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)

        s = comp_obj["feature_slices"]
        # 6채널 모두
        slices = [
            ("low_R",  "R_Low"),
            ("low_G",  "G_Low"),
            ("low_B",  "B_Low"),
            ("high_R", "R_High"),
            ("high_G", "G_High"),
            ("high_B", "B_High"),
        ]

        Phi = self._stack_basis(knots, L=256)    # (256,K)

        A_blocks = []
        for key_slice, _pretty_name in slices:
            sl = slice(s[key_slice][0], s[key_slice][1])   # e.g. (0,33), (33,66), ...
            beta = coef[sl] / np.maximum(scale[sl], 1e-12)  # (K,)
            A_ch = Phi * beta.reshape(1, -1)                # (256,K)
            A_blocks.append(A_ch)

        A = np.hstack(A_blocks).astype(np.float32)          # (256, 6K)
        logging.info(f"[Jacobian] {comp} A 행렬 shape: {A.shape}") 
        return A
    
    def _load_prediction_models(self):
        """
        hybrid_*_model.pkl 파일들을 불러와서 self.models_Y0_bundle에 저장.
        (Gamma / Cx / Cy)
        """
        model_names = {
            "Gamma": "hybrid_Gamma_model.pkl",
            "Cx": "hybrid_Cx_model.pkl",
            "Cy": "hybrid_Cy_model.pkl",
        }

        models_dir = cf.get_normalized_path(__file__, '.', 'models')
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
        return bundle
    
    def _set_vac_active(self, enable: bool) -> bool:
        try:
            logging.debug("현재 VAC 적용 상태를 확인합니다.")
            current_status = self._check_vac_status()
            current_active = bool(current_status.get("activated", False))

            if current_active == enable:
                logging.info(f"VAC already {'ON' if enable else 'OFF'} - skipping command.")
                return True

            self.send_command(self.ser_tv, 's')
            cmd = (
                "luna-send -n 1 -f "
                "luna://com.webos.service.panelcontroller/setVACActive "
                f"'{{\"OnOff\":{str(enable).lower()}}}'"
            )
            self.send_command(self.ser_tv, cmd)
            self.send_command(self.ser_tv, 'exit')
            time.sleep(0.5)
            st = self._check_vac_status()
            return bool(st.get("activated", False)) == enable
        
        except Exception as e:
            logging.error(f"VAC {'ON' if enable else 'OFF'} 전환 실패: {e}")
            return False
        
    def _check_vac_status(self):
        self.send_command(self.ser_tv, 's')
        getVACSupportstatus = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/getVACSupportStatus \'{"subscribe":true}\''
        VAC_support_status = self.send_command(self.ser_tv, getVACSupportstatus)
        VAC_support_status = self.extract_json_from_luna_send(VAC_support_status)
        self.send_command(self.ser_tv, 'exit')
        
        if not VAC_support_status:
            logging.warning("Failed to retrieve VAC support status from TV.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        if not VAC_support_status.get("isSupport", False):
            logging.info("VAC is not supported on this model.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        activated = VAC_support_status.get("isActivated", False)
        logging.info(f"VAC 적용 상태: {activated}")
                
        return {"supported": True, "activated": activated}
        
    def _dev_zero_lut_from_file(self):
        """원본 VAC JSON을 골라 6개 LUT 키만 0으로 덮어쓴 JSON을 임시파일로 저장하고 자동으로 엽니다."""
        # 1) 원본 JSON 선택
        fname, _ = QFileDialog.getOpenFileName(
            self, "원본 VAC JSON 선택", "", "JSON Files (*.json);;All Files (*)"
        )
        if not fname:
            return

        try:
            # 2) 순서 보존 로드
            with open(fname, "r", encoding="utf-8") as f:
                raw_txt = f.read()
            vac_dict = json.loads(raw_txt, object_pairs_hook=OrderedDict)

            # 3) LUT 6키를 모두 0으로 구성 (4096 포인트)
            zeros = np.zeros(4096, dtype=np.int32)
            zero_luts = {
                "RchannelLow":  zeros,
                "RchannelHigh": zeros,
                "GchannelLow":  zeros,
                "GchannelHigh": zeros,
                "BchannelLow":  zeros,
                "BchannelHigh": zeros,
            }

            vac_text = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=zero_luts)

            # 5) 임시파일로 저장
            fd, tmp_path = tempfile.mkstemp(prefix="VAC_zero_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(vac_text)

            # 6) startfile
            try:
                os.startfile(tmp_path)
            except Exception as e:
                logging.warning(f"임시파일 자동 열기 실패: {e}")

            QMessageBox.information(self, "완료", f"Zero-LUT JSON 임시파일 생성 및 열기 완료:\n{tmp_path}")

        except Exception as e:
            logging.exception(e)
            QMessageBox.critical(self, "오류", f"처리 중 오류: {e}")
        
    def build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict = None) -> str:
        """
        base_vac_dict : TV에서 읽은 원본 JSON(dict; 키 순서 유지 권장)
        new_lut_tvkeys: 교체할 LUT만 전달 시 병합 (TV 원 키명 그대로)
                        {"RchannelLow":[...4096], "RchannelHigh":[...], ...}
        return: TV에 바로 쓸 수 있는 탭 포맷 문자열
        """
        from collections import OrderedDict
        import numpy as np, json

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

        # -------------------------------
        # 포맷터
        # -------------------------------
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

        # -------------------------------
        # 본문 생성
        # -------------------------------
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
    
    def _run_off_baseline_then_on(self):
        profile_off = SessionProfile(
            legend_text="VAC OFF (Ref.)",
            cie_label="data_1",
            table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
            ref_store=None
        )

        def _after_off(store_off):
            self._off_store = store_off
            # logging.debug(f"VAC OFF 측정 결과:\n{self._off_store}")
            self._step_done(1)
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
        self._step_start(2)
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 최적화 루프 종료")
            return
        vac_data_dict = json.loads(vac_data)
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in vac_data_dict.items() if "channel" in key
        }
        self._update_lut_chart_and_table(lut_dict_plot)
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
                # logging.debug(f"VAC ON 측정 결과:\n{self._on_store}")
                
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
        
    def _debug_log_knot_update(self, iter_idx, knots, delta_h, lut256_before, lut256_after):
        """
        iter_idx        : 현재 iteration 번호 (1, 2, ...)
        knots           : self._jac_artifacts["knots"]  # 길이 K, 예: [0,8,16,...,255]
        delta_h         : (6K,) 이번 iteration에서 solve한 Δh
        lut256_before   : dict of 6채널 256길이 LUT (보정 전, float32)
        lut256_after    : dict of 6채널 256길이 LUT (보정 후, float32)

        이걸 로그에 예쁘게 찍어 분석용으로 쓸 수 있게 해 준다.
        """
        try:
            K = len(knots)
            # 채널 분해
            dh_RL = delta_h[0*K : 1*K]
            dh_GL = delta_h[1*K : 2*K]
            dh_BL = delta_h[2*K : 3*K]
            dh_RH = delta_h[3*K : 4*K]
            dh_GH = delta_h[4*K : 5*K]
            dh_BH = delta_h[5*K : 6*K]

            def _summ(ch_name, dh_vec):
                # dh_vec 길이 K
                # 상위 몇 개만 큰 변화 순으로 보여주면 어디가 움직였는지 직관적으로 파악 가능
                mag = np.abs(dh_vec)
                top_idx = np.argsort(mag)[::-1][:5]  # 변화량 큰 상위 5개 knot
                msg_lines = [f"    {ch_name} top5 |knot(gray)->Δh|:"]
                for i in top_idx:
                    msg_lines.append(
                        f"      knot#{i:02d} (gray≈{knots[i]:3d}) : Δh={dh_vec[i]:+.4f}"
                    )
                return "\n".join(msg_lines)

            logging.info("======== [CORR DEBUG] Iter %d Knot Δh ========\n%s\n%s\n%s\n%s\n%s\n%s",
                iter_idx,
                _summ("R_Low ", dh_RL),
                _summ("G_Low ", dh_GL),
                _summ("B_Low ", dh_BL),
                _summ("R_High", dh_RH),
                _summ("G_High", dh_GH),
                _summ("B_High", dh_BH),
            )

            # LUT 전/후 차이도 간단 비교 (예: High 채널만 대표로)
            def _lut_diff_stats(name):
                before = np.asarray(lut256_before[name], dtype=np.float32)
                after  = np.asarray(lut256_after[name],  dtype=np.float32)
                diff   = after - before
                return (float(np.min(diff)),
                        float(np.max(diff)),
                        float(np.mean(diff)),
                        float(np.std(diff)))

            for ch in ["R_Low","G_Low","B_Low","R_High","G_High","B_High"]:
                dmin, dmax, dmean, dstd = _lut_diff_stats(ch)
                logging.debug(
                    "[CORR DEBUG] Iter %d %s LUT256 delta stats: "
                    "min=%+.4f max=%+.4f mean=%+.4f std=%.4f",
                    iter_idx, ch, dmin, dmax, dmean, dstd
                )

        except Exception:
            logging.exception("[CORR DEBUG] knot update logging failed")
            
    def _smooth_and_monotone(self, arr, win=9):
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

    def _fix_low_high_order(self, low_arr, high_arr):
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

    def _nudge_midpoint(self, low_arr, high_arr, max_err=3.0, strength=0.5):
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

    def _finalize_channel_pair_safely(self, low_arr, high_arr):
        """
        마지막 안전화 단계:
        1) 다시 Low>High 방지
        2) 단조 증가 강제 (_enforce_monotone)
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
        low  = self._enforce_monotone(low)
        high = self._enforce_monotone(high)

        # (3) 엔드포인트 고정
        low[0]  = 0.0
        high[0] = 0.0
        low[-1]  = 4095.0
        high[-1] = 4095.0

        # (4) clip
        low  = np.clip(low ,  0.0, 4095.0)
        high = np.clip(high, 0.0, 4095.0)

        return low.astype(np.float32), high.astype(np.float32)
            
    def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
        logging.info(f"[CORR] iteration {iter_idx} start")
        self._step_start(2)

        # 1) 현재 TV LUT (캐시) 확보
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.warning("[CORR] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
            return None
        vac_dict = self._vac_dict_cache  # TV에서 읽어온 최신 VAC JSON (4096포인트, 12bit)

        # 2) 4096 → 256 다운샘플 (Low/High 전채널)
        vac_lut_4096 = {
            "R_Low":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "R_High": np.asarray(vac_dict["RchannelHigh"], dtype=np.float32),
            "G_Low":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "G_High": np.asarray(vac_dict["GchannelHigh"], dtype=np.float32),
            "B_Low":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "B_High": np.asarray(vac_dict["BchannelHigh"], dtype=np.float32),
        }

        lut256 = {
            "R_Low":  self._down4096_to_256(vac_lut_4096["R_Low"]),
            "G_Low":  self._down4096_to_256(vac_lut_4096["G_Low"]),
            "B_Low":  self._down4096_to_256(vac_lut_4096["B_Low"]),
            "R_High": self._down4096_to_256(vac_lut_4096["R_High"]),
            "G_High": self._down4096_to_256(vac_lut_4096["G_High"]),
            "B_High": self._down4096_to_256(vac_lut_4096["B_High"]),
        }
        # lut256[...] 은 여전히 0~4095 스케일 (12bit 값) 상태입니다.
        lut256_before = {k: v.copy() for k, v in lut256.items()}

        # 3) Δ 목표(white/main 기준): OFF vs ON 차이
        #    Gamma: 1..254 유효, Cx/Cy: 0..255
        d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
        # d_targets = {"Gamma":(256,), "Cx":(256,), "Cy":(256,)}, 값 = (ON - OFF)

        # 아주 작은 오차(이미 충분히 맞은 gray)는 굳이 고치지 말자 → 0으로
        thr_c = 0.003
        thr_gamma = 0.03
        for g in range(256):
            if (
                abs(d_targets["Cx"][g]) <= thr_c and
                abs(d_targets["Cy"][g]) <= thr_c and
                abs(d_targets["Gamma"][g]) <= thr_gamma
            ):
                d_targets["Cx"][g]    = 0.0
                d_targets["Cy"][g]    = 0.0
                d_targets["Gamma"][g] = 0.0

        # 4) 결합 선형계
        #    ΔY ≈ [A_Gamma; A_Cx; A_Cy] · Δh
        #    여기서 A_* shape = (256, 6K). Δh shape = (6K,)
        #    wG, wCx, wCy는 가중치
        wCx = 0.05
        wCy = 0.5
        wG  = 1.0

        A_cat = np.vstack([
            wG  * self.A_Gamma,
            wCx * self.A_Cx,
            wCy * self.A_Cy
        ]).astype(np.float32)  # (256*3, 6K)

        b_cat = -np.concatenate([
            wG  * d_targets["Gamma"],
            wCx * d_targets["Cx"],
            wCy * d_targets["Cy"]
        ]).astype(np.float32)  # (256*3,)

        # 유효치 마스크(특히 gamma의 NaN 등에서 온 0/inf 제거)
        mask = np.isfinite(b_cat)
        A_use = A_cat[mask, :]  # (M, 6K)
        b_use = b_cat[mask]     # (M,)

        # 5) 리지 회귀 해 (Δh) 구하기
        #    (A^T A + λI) Δh = A^T b
        ATA = A_use.T @ A_use            # (6K,6K)
        rhs = A_use.T @ b_use            # (6K,)
        ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
        delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)  # (6K,)

        # --------- 디버그 준비 -------------
        K = len(self._jac_artifacts["knots"])
        knots = np.asarray(self._jac_artifacts["knots"], dtype=np.int32)

        # 6) knot delta → per-gray 보정곡선(256포인트)로 전개
        #    delta_h 해석:
        #    [R_Low_knots(0:K),
        #     G_Low_knots(K:2K),
        #     B_Low_knots(2K:3K),
        #     R_High_knots(3K:4K),
        #     G_High_knots(4K:5K),
        #     B_High_knots(5K:6K)]
        Phi = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)

        idx0 = 0
        dh_RL = delta_h[idx0      : idx0+K]; idx0 += K
        dh_GL = delta_h[idx0      : idx0+K]; idx0 += K
        dh_BL = delta_h[idx0      : idx0+K]; idx0 += K
        dh_RH = delta_h[idx0      : idx0+K]; idx0 += K
        dh_GH = delta_h[idx0      : idx0+K]; idx0 += K
        dh_BH = delta_h[idx0      : idx0+K]

        corr_RL = Phi @ dh_RL  # (256,)
        corr_GL = Phi @ dh_GL
        corr_BL = Phi @ dh_BL
        corr_RH = Phi @ dh_RH
        corr_GH = Phi @ dh_GH
        corr_BH = Phi @ dh_BH

        # 7) 1차 LUT 후보 (아직 후처리 전)
        lut256_new = {
            "R_Low":  (lut256["R_Low"]  + corr_RL).astype(np.float32),
            "G_Low":  (lut256["G_Low"]  + corr_GL).astype(np.float32),
            "B_Low":  (lut256["B_Low"]  + corr_BL).astype(np.float32),
            "R_High": (lut256["R_High"] + corr_RH).astype(np.float32),
            "G_High": (lut256["G_High"] + corr_GH).astype(np.float32),
            "B_High": (lut256["B_High"] + corr_BH).astype(np.float32),
        }

        # =========================
        # ▼ NEW: 안전 후처리 파이프라인
        # =========================
        #
        # 목적:
        #   - 톱니/지글지글 완화 (moving average + monotone)
        #   - Low > High 금지
        #   - (Low+High)/2 가 이상하게 튀는 gray에서만 살짝 눌러서
        #     감마/휘도 급튜는 구간 줄이기
        #   - g=0은 항상 0, g=255는 항상 4095
        #   - 최종적으로 다시 monotone + clip

        for ch in ("R", "G", "B"):
            Lk = f"{ch}_Low"
            Hk = f"{ch}_High"

            # (0) 엔드포인트를 미리 합리적으로 잡아준다.
            lut256_new[Lk][0]   = 0.0
            lut256_new[Hk][0]   = 0.0
            lut256_new[Lk][255] = 4095.0
            lut256_new[Hk][255] = 4095.0

            # (1) Low/High 역전 금지 (1차 정리)
            low_fixed, high_fixed = self._fix_low_high_order(
                lut256_new[Lk], lut256_new[Hk]
            )

            # (2) 스무딩 + 단조 증가 보장으로 톱니 제거
            low_smooth  = self._smooth_and_monotone(low_fixed,  win=9)
            high_smooth = self._smooth_and_monotone(high_fixed, win=9)

            # (3) 평균 밝기(mid) 너무 이상하게 튀는 지점만 살짝 눌러주기
            low_mid, high_mid = self._nudge_midpoint(
                low_smooth, high_smooth,
                max_err=3.0,    # 12bit에서 ±3카운트 이상 벗어나면만 관여
                strength=0.5    # 그 오차의 절반만 교정
            )

            # (4) 최종 안전화:
            #     - 다시 Low<=High
            #     - 단조 재보장
            #     - 0/255 엔드포인트 고정
            #     - clip(0..4095)
            low_final, high_final = self._finalize_channel_pair_safely(
                low_mid, high_mid
            )

            lut256_new[Lk] = low_final
            lut256_new[Hk] = high_final

        # 이제 lut256_new[*] 는
        #  - 단조 증가
        #  - Low <= High
        #  - g=0 → 0, g=255 → 4095
        #  - 고주파 톱니 줄어듦
        # =========================
        # ▲ NEW 파이프라인 끝
        # =========================

        # --------- 디버그 로그 (보정량 요약) -------------
        try:
            self._debug_log_knot_update(
                iter_idx=iter_idx,
                knots=knots,
                delta_h=delta_h,
                lut256_before=lut256_before,
                lut256_after=lut256_new,
            )
        except Exception:
            logging.exception("[CORR DEBUG] _debug_log_knot_update failed")
        # -------------------------------------------------

        # 8) 256 → 4096 업샘플 (모든 채널), 정수화
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

        # 9) UI용 차트/테이블 갱신
        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],
            "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],
            "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],
            "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        # 10) TV에 쓰고, 다시 읽고, 다시 측정 → 스펙 체크 흐름은 기존 그대로
        def _after_write(ok, msg):
            logging.info(f"[VAC Write] {msg}")
            if not ok:
                return
            # 쓰기 성공 → 재읽기
            logging.info("보정 LUT TV Reading 시작")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            if not vac_dict_after:
                logging.error("보정 후 VAC 재읽기 실패")
                return
            logging.info("보정 LUT TV Reading 완료")
            self._step_done(3)

            # 1) 캐시/차트 갱신
            self._vac_dict_cache = vac_dict_after

            # 2) ON 시리즈 리셋 (OFF는 참조 유지)
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            # 3) 보정 후(=ON) 측정 세션 시작
            profile_corr = SessionProfile(
                legend_text=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={
                    "lv":4, "cx":5, "cy":6, "gamma":7,
                    "d_cx":8, "d_cy":9, "d_gamma":10
                },
                ref_store=self._off_store  # 항상 OFF 대비 Δ
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr  # 최신 ON(보정 후) 측정 결과

                self._step_start(5)
                self._spec_thread = SpecEvalThread(
                    self._off_store, self._on_store,
                    thr_gamma=0.05, thr_c=0.003, parent=self
                )
                self._spec_thread.finished.connect(
                    lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx, max_iters)
                )
                self._spec_thread.start()

            logging.info("보정 LUT 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_corr,
                gray_levels=getattr(op, "gray_levels_256", list(range(256))),
                gamma_patterns=('white',),             # white만 측정
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                cs_settle_ms=1000,
                on_done=_after_corr
            )

        logging.info("LUT {}차 보정 완료".format(iter_idx))
        logging.info("LUT {}차 TV Writing 시작".format(iter_idx))

        # 11) VAC JSON 재조립 후 TV에 write
        vac_write_json = self.build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)
            
    def _check_spec_pass(self, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
    # white/main만 기준
        def _extract_white(series_store):
            lv = np.zeros(256); cx = np.zeros(256); cy = np.zeros(256)
            for g in range(256):
                tup = series_store['gamma']['main']['white'].get(g, None)
                if tup: lv[g], cx[g], cy[g] = tup
                else:   lv[g]=np.nan; cx[g]=np.nan; cy[g]=np.nan
            return lv, cx, cy

        lv_ref, cx_ref, cy_ref = _extract_white(off_store)
        lv_on , cx_on , cy_on  = _extract_white(on_store)

        G_ref = self._compute_gamma_series(lv_ref)
        G_on  = self._compute_gamma_series(lv_on)

        dG  = np.abs(G_on - G_ref)
        dCx = np.abs(cx_on - cx_ref)
        dCy = np.abs(cy_on - cy_ref)

        max_dG  = np.nanmax(dG)
        max_dCx = np.nanmax(dCx)
        max_dCy = np.nanmax(dCy)

        logging.info(f"[SPEC] max|ΔGamma|={max_dG:.6f} (≤{thr_gamma}), max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c})")
        return (max_dG <= thr_gamma) and (max_dCx <= thr_c) and (max_dCy <= thr_c)

    def _build_delta_targets_from_stores(self, off_store, on_store):
        # Δ = (ON - OFF). white/main
        lv_ref, cx_ref, cy_ref = np.zeros(256), np.zeros(256), np.zeros(256)
        lv_on , cx_on , cy_on  = np.zeros(256), np.zeros(256), np.zeros(256)
        for g in range(256):
            tR = off_store['gamma']['main']['white'].get(g, None)
            tO = on_store['gamma']['main']['white'].get(g, None)
            if tR: lv_ref[g], cx_ref[g], cy_ref[g] = tR
            else:  lv_ref[g]=np.nan; cx_ref[g]=np.nan; cy_ref[g]=np.nan
            if tO: lv_on[g], cx_on[g], cy_on[g] = tO
            else:  lv_on[g]=np.nan; cx_on[g]=np.nan; cy_on[g]=np.nan

        G_ref = self._compute_gamma_series(lv_ref)
        G_on  = self._compute_gamma_series(lv_on)
        d = {
            "Gamma": (G_on - G_ref),
            "Cx":    (cx_on - cx_ref),
            "Cy":    (cy_on - cy_ref),
        }
        # NaN → 0 (선형계 마스킹에서도 걸러지니 안정성↑)
        for k in d:
            d[k] = np.nan_to_num(d[k], nan=0.0).astype(np.float32)
        return d
    
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

        # 현재 세션이 OFF 레퍼런스인지, ON/보정 런인지 상태 문자열 결정
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        # 두 역할을 results 키로 직접 순회 (측정기 객체 비교 X)
        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                # 측정 실패/결측인 경우
                store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
                continue

            x, y, lv, cct, duv = res
            # 스토어 업데이트 (white 테이블/감마 계산 등에 사용)
            store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

            # ▶▶ 차트 업데이트 (간소화 API)
            # GammaChartVAC: add_point(state, role, pattern, gray, luminance)
            self.vac_optimization_gamma_chart.add_point(
                state=state,
                role=role,               # 'main' 또는 'sub'
                pattern=pattern,         # 'white'|'red'|'green'|'blue'
                gray=int(gray),
                luminance=float(lv)
            )

        if pattern == 'white':
            is_on_session = (profile.ref_store is not None)
            if is_on_session:
                ok_now = self._is_gray_spec_ok(gray, thr_gamma=0.05, thr_c=0.003)
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
                if ref_main is not None:
                    _, cx_r, cy_r = ref_main
                    if np.isfinite(cx_m):
                        d_cx = cx_m - cx_r
                        self._set_item_with_spec(
                            table_inst1, gray, cols['d_cx'], f"{d_cx:.6f}",
                            is_spec_ok=(abs(d_cx) <= 0.003)  # thr_c
                        )
                    if np.isfinite(cy_m):
                        d_cy = cy_m - cy_r
                        self._set_item_with_spec(
                            table_inst1, gray, cols['d_cy'], f"{d_cy:.6f}",
                            is_spec_ok=(abs(d_cy) <= 0.003)  # thr_c
                        )

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
                
    def _is_gray_spec_ok(self, gray:int, *, thr_gamma=0.05, thr_c=0.003) -> bool:
        # OFF 레퍼런스
        ref = self._off_store['gamma']['main']['white'].get(gray, None)
        on  = self._on_store ['gamma']['main']['white'].get(gray, None)
        if not ref or not on:
            return True  # 데이터 없으면 패스 취급(측정 실패는 상위 로직에서 처리)
        lv_r, cx_r, cy_r = ref
        lv_o, cx_o, cy_o = on

        # 감마 한 점 계산(안전 가드: 전체 벡터 재계산보다 간단 추정)
        # 정확도를 높이려면 기존 _compute_gamma_series로 전체 재계산 후 gray 인덱스 꺼내도 됩니다.
        # 여기서는 간단화를 위해 _compute_gamma_series 사용:
        def _one_gamma(store):
            lv = np.zeros(256); 
            for g in range(256):
                t = store['gamma']['main']['white'].get(g, None)
                lv[g] = float(t[0]) if t else np.nan
            return self._compute_gamma_series(lv)

        G_ref = _one_gamma(self._off_store)
        G_on  = _one_gamma(self._on_store)
        dG  = abs(G_on[gray]) if np.isfinite(G_on[gray]) and np.isfinite(G_ref[gray]) else 0.0
        dCx = abs(cx_o - cx_r) if np.isfinite(cx_o) and np.isfinite(cx_r) else 0.0
        dCy = abs(cy_o - cy_r) if np.isfinite(cy_o) and np.isfinite(cy_r) else 0.0

        return (dG <= thr_gamma) and (dCx <= thr_c) and (dCy <= thr_c)
        
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
            # 세션 재개: 다음 gray로 자연 진행되게끔 g_idx는 기존 루프가 제어
            self._sess['_gray_fix'] = None
            self._resume_session()
            return

        ctx['tries'] = tries + 1
        logging.info(f"[GRAY-FIX] g={g} try={ctx['tries']}/{maxr}")

        # ===== 1) Δ 타깃(해당 g만) =====
        def _get_off_on_xyG(store_off, store_on, gray):
            # xy/lv 추출
            tR = store_off['gamma']['main']['white'].get(gray, None)
            tO = store_on ['gamma']['main']['white'].get(gray, None)
            lv_r, cx_r, cy_r = (tR if tR else (np.nan, np.nan, np.nan))
            lv_o, cx_o, cy_o = (tO if tO else (np.nan, np.nan, np.nan))
            # 감마는 전체에서 계산 후 해당 g만 취함
            G_ref = self._compute_gamma_series(
                np.array([store_off['gamma']['main']['white'].get(i,(np.nan,)*3)[0] for i in range(256)], float)
            )
            G_on  = self._compute_gamma_series(
                np.array([store_on ['gamma']['main']['white'].get(i,(np.nan,)*3)[0] for i in range(256)], float)
            )
            return (G_on[gray]-G_ref[gray], cx_o-cx_r, cy_o-cy_r)

        dG, dCx, dCy = _get_off_on_xyG(self._off_store, self._on_store, g)

        # 소소한 deadband: 이미 충분히 작으면 바로 재측정으로 넘어가도 됨
        if (abs(dG) <= thr_gamma) and (abs(dCx) <= thr_c) and (abs(dCy) <= thr_c):
            logging.info(f"[GRAY-FIX] g={g} already within thr (skip fix) → remeasure")
            return self._remeasure_same_gray(g)

        # ===== 2) 자코비안 g행 구성 (결합 가중치는 기존과 동일)
        wG, wCx, wCy = 1.0, 0.05, 0.5  # 필요 시 UI/설정으로
        Ag = np.vstack([
            wG  * self.A_Gamma[g:g+1, :],   # (1,6K)
            wCx * self.A_Cx   [g:g+1, :],
            wCy * self.A_Cy   [g:g+1, :],
        ])                                  # (3,6K)
        b  = -np.array([wG*dG, wCx*dCx, wCy*dCy], dtype=np.float32)  # (3,)

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

    def _remeasure_same_gray(self, gray:int):
        """같은 g를 즉시 재측정한다(white/main만)."""
        # 차트/테이블에서 ON 시리즈에 덮어쓰도록 그대로 측정 루틴 재사용
        # 단, 세션은 여전히 paused 상태. g_idx는 증가시키지 않음.
        self.changeColor(f"{gray},{gray},{gray}")
        # settle 후 한 페어 측정만 트리거
        def done_pair(pattern, g):
            # 기존 핸들러를 재사용하되, g_idx 증가는 막아야 함 (아래 4번 패치 참고)
            self._trigger_gamma_pair(pattern='white', gray=g)
        QTimer.singleShot(self._sess.get('cs_settle_ms', 1000), lambda: done_pair('white', gray))
    
                
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

    def _set_item(self, table, row, col, value):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))

        table.scrollToItem(item, QAbstractItemView.PositionAtTop)
        
    def _set_item_with_spec(self, table, row, col, value, *, is_spec_ok: bool):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))
        # 🔸 스펙 OUT만 빨간 배경
        if is_spec_ok:
            item.setBackground(QColor(0, 0, 255))  # 기본(흰색)로 돌림
        else:
            item.setBackground(QColor(255, 0, 0))  # 연한 빨강

        table.scrollToItem(item, QAbstractItemView.PositionAtCenter)

    def _compute_gamma_series(self, lv_vec_256):
        lv = np.asarray(lv_vec_256, dtype=np.float64)
        gamma = np.full(256, np.nan, dtype=np.float64)
        lv0 = lv[0]
        denom = np.max(lv[1:] - lv0)
        if not np.isfinite(denom) or denom <= 0:
            return gamma
        nor = (lv - lv0) / denom
        gray = np.arange(256, dtype=np.float64)
        gray_norm = gray / 255.0
        valid = (gray >= 1) & (gray <= 254) & (nor > 0) & np.isfinite(nor)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma[valid] = np.log(nor[valid]) / np.log(gray_norm[valid])
        return gamma
    
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

    def _down4096_to_256(self, arr4096):
        arr4096 = np.asarray(arr4096, dtype=np.float32)
        idx = np.round(np.linspace(0, 4095, 256)).astype(int)
        return arr4096[idx]

    def _up256_to_4096(self, arr256):
        arr256 = np.asarray(arr256, dtype=np.float32)
        x_small = np.linspace(0, 1, 256)
        x_big   = np.linspace(0, 1, 4096)
        return np.interp(x_big, x_small, arr256).astype(np.float32)

    def _enforce_monotone(self, arr):
        # 제자리 누적 최대치
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                arr[i] = arr[i-1]
        return arr
        
    def _fetch_vac_by_model(self, panel_maker, frame_rate):
        """
        DB: W_VAC_Application_Status에서 Panel_Maker/Frame_Rate 매칭 → VAC_Info_PK 얻고
            W_VAC_Info.PK=VAC_Info_PK → VAC_Data 읽어서 반환
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
            self._update_spec_views(iter_idx, self._off_store, self._on_store)

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
            
    def _update_spec_views(self, iter_idx, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
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

    def debug_check_prediction_contract_once(self):
        """
        - DB LUT(또는 캐시 LUT)를 4096→256 다운샘플해 계약대로 X를 만들고
        - 각 모델의 n_features, 예측 결과 통계(평균/표준편차)를 로그로 확인
        - g=128 한 줄의 피처를 상세 출력
        """
        # 1) 현재 사용할 LUT 소스 확보 (DB 읽은 것 또는 예측 LUT)
        if hasattr(self, "_vac_dict_cache") and self._vac_dict_cache:
            src = self._vac_dict_cache
        elif hasattr(self, "_vac_dict_last_preview") and self._vac_dict_last_preview:
            src = self._vac_dict_last_preview
        else:
            logging.error("[Predict/Debug] No LUT source available (need _vac_dict_cache or _vac_dict_last_preview).")
            return

        lut256 = {
            "R_Low":  self._down4096_to_256(src["RchannelLow"]),
            "R_High": self._down4096_to_256(src["RchannelHigh"]),
            "G_Low":  self._down4096_to_256(src["GchannelLow"]),
            "G_High": self._down4096_to_256(src["GchannelHigh"]),
            "B_Low":  self._down4096_to_256(src["BchannelLow"]),
            "B_High": self._down4096_to_256(src["BchannelHigh"]),
        }

        panel, fr, my = self._get_ui_meta()
        X, contract = self._build_feature_matrix_W_checked(
            lut256, panel_text=panel, frame_rate=fr, model_year=my
        )

        # 2) 예측하고 통계 로그
        def _pred(payload):
            base = payload["linear_model"].predict(X).astype(np.float32)
            resid= payload["rf_residual"].predict(X).astype(np.float32)
            mu   = float(payload["target_scaler"]["mean"])
            sd   = float(payload["target_scaler"]["std"])
            return (base + resid) * sd + mu

        for comp in ("Gamma","Cx","Cy"):
            y = _pred(self.models_Y0_bundle[comp])
            logging.debug(f"[Predict/Debug] {comp}: mean={np.nanmean(y):.6g}, std={np.nanstd(y):.6g}, min={np.nanmin(y):.6g}, max={np.nanmax(y):.6g}")

        # 3) g=128 한 줄 피처 상세
        g = 128
        logging.debug(f"[Predict/Debug] g={g} feature row: {X[g,:].tolist()}")
        
    # ===== [ADD] 패널 원핫 =====
    def _panel_onehot(self, panel_text: str):
        # 학습 때 쓰던 순서와 동일해야 합니다.
        PANEL_MAKER_CATEGORIES = ['HKC(H2)', 'HKC(H5)', 'BOE', 'CSOT', 'INX']
        v = np.zeros(len(PANEL_MAKER_CATEGORIES), np.float32)
        try:
            i = PANEL_MAKER_CATEGORIES.index(panel_text)
            v[i] = 1.0
        except ValueError:
            # 미스매치면 전부 0 (학습과 계약 유지)
            pass
        return v
    
    # ===== [ADD] per-gray(W) 한 행 피처 (길이=18) =====
    def _build_runtime_feature_row_W(
        self,
        lut256_norm: dict,
        gray: int,
        *,
        panel_text: str,
        frame_rate: float,
        model_year_2digit: float = None,
        model_year: float = None,   # ← alias 허용
    ):
        # --- alias 정리 ---
        if model_year_2digit is None:
            if model_year is not None:
                model_year_2digit = float(int(model_year) % 100)
            else:
                model_year_2digit = 0.0

        row = [
            float(lut256_norm['R_Low'][gray]),  float(lut256_norm['R_High'][gray]),
            float(lut256_norm['G_Low'][gray]),  float(lut256_norm['G_High'][gray]),
            float(lut256_norm['B_Low'][gray]),  float(lut256_norm['B_High'][gray]),
        ]
        row.extend(self._panel_onehot(panel_text).tolist())
        row.append(float(frame_rate))
        row.append(float(model_year_2digit))   # 두 자리 확정
        row.append(gray / 255.0)               # gray_norm
        row.extend([1.0, 0.0, 0.0, 0.0])       # W one-hot
        return np.asarray(row, dtype=np.float32)

    def _predict_Y0W_from_models(self, lut256_dict, *, panel_text, frame_rate, model_year):
        """
        저장된 hybrid_*_model.pkl 3개로 'W' 패턴 256 포인트의 (Gamma, Cx, Cy) 예측 벡터를 생성
        """
        # ✅ LUT는 반드시 0..1 스케일로 맞춘다
        def _norm01(a): 
            return np.clip(np.asarray(a, np.float32) / 4095.0, 0.0, 1.0)
        lut256_norm = {
            "R_Low":  _norm01(lut256_dict["R_Low"]),
            "R_High": _norm01(lut256_dict["R_High"]),
            "G_Low":  _norm01(lut256_dict["G_Low"]),
            "G_High": _norm01(lut256_dict["G_High"]),
            "B_Low":  _norm01(lut256_dict["B_Low"]),
            "B_High": _norm01(lut256_dict["B_High"]),
        }

        # 256행 피처 매트릭스
        # ✅ _build_runtime_feature_row_W의 파라미터명은 model_year_2digit 입니다.
        X_rows = [ self._build_runtime_feature_row_W(
                        lut256_norm, g,
                        panel_text=panel_text,
                        frame_rate=frame_rate,
                        model_year_2digit=float(model_year)  # 두 자리 숫자 가정
                ) for g in range(256) ]
        X = np.vstack(X_rows).astype(np.float32)

        def _pred_one(payload):
            lin = payload["linear_model"]; rf = payload["rf_residual"]
            tgt = payload["target_scaler"]; y_mean = float(tgt["mean"]); y_std = float(tgt["std"])
            base_s  = lin.predict(X).astype(np.float32)
            resid_s = rf.predict(X).astype(np.float32)
            y = (base_s + resid_s) * y_std + y_mean
            return y.astype(np.float32)

        yG  = _pred_one(self.models_Y0_bundle["Gamma"])
        yCx = _pred_one(self.models_Y0_bundle["Cx"])
        yCy = _pred_one(self.models_Y0_bundle["Cy"])

        # Gamma 0/255는 NaN
        yG[0] = np.nan; yG[255] = np.nan
        return {"Gamma": yG, "Cx": yCx, "Cy": yCy}

    def _delta_targets_vs_OFF_from_pred(self, y_pred_W, off_store):
        """
        OFF 실측(white/main)과 예측 ON 값을 비교해 Δ 타깃(길이 256)을 만든다.
        """
        # OFF store → lv, cx, cy 시리즈
        lv_ref = np.zeros(256); cx_ref = np.zeros(256); cy_ref = np.zeros(256)
        for g in range(256):
            tR = off_store['gamma']['main']['white'].get(g, None)
            if tR: lv_ref[g], cx_ref[g], cy_ref[g] = tR
            else:  lv_ref[g]=np.nan; cx_ref[g]=np.nan; cy_ref[g]=np.nan
        G_ref = self._compute_gamma_series(lv_ref)

        d = {
            "Gamma": (np.nan_to_num(y_pred_W["Gamma"], nan=np.nan) - G_ref),
            "Cx":    (y_pred_W["Cx"] - cx_ref),
            "Cy":    (y_pred_W["Cy"] - cy_ref),
        }
        for k in d:
            d[k] = np.nan_to_num(d[k], nan=0.0).astype(np.float32)
        return d

    def _down4096_to_256_float(self, arr4096):
        idx = np.round(np.linspace(0, 4095, 256)).astype(int)
        return np.asarray(arr4096, dtype=np.float32)[idx]

    def _predictive_first_optimize(self, vac_data_json, *, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3):
        """
        DB에서 가져온 VAC JSON을 예측모델+자코비안으로 미리 n회 보정.
        - 감마 정확도 낮음: wG(기본 0.4)로 영향 축소
        - return: (optimized_vac_json_str, lut_dict_4096)  혹은 (None, None) 실패 시
        """
        try:
            vac_dict = json.loads(vac_data_json)

            # 1️⃣ 기존 그대로 — 4096→256 다운샘플 (12bit 값 그대로)
            lut256 = {
                "R_Low":  self._down4096_to_256_float(vac_dict["RchannelLow"]),
                "R_High": self._down4096_to_256_float(vac_dict["RchannelHigh"]),
                "G_Low":  self._down4096_to_256_float(vac_dict["GchannelLow"]),
                "G_High": self._down4096_to_256_float(vac_dict["GchannelHigh"]),
                "B_Low":  self._down4096_to_256_float(vac_dict["BchannelLow"]),
                "B_High": self._down4096_to_256_float(vac_dict["BchannelHigh"]),
            }

            if not hasattr(self, "A_Gamma"):
                logging.error("[PredictOpt] Jacobian not prepared.")
                return None, None

            panel, fr, model_year = self._get_ui_meta()

            K   = len(self._jac_artifacts["knots"])
            Phi = self._stack_basis(self._jac_artifacts["knots"])

            # 이 변수들은 계속 12bit 스케일 유지
            high_R = lut256["R_High"].copy()
            high_G = lut256["G_High"].copy()
            high_B = lut256["B_High"].copy()

            for it in range(1, n_iters + 1):
                # ✅ 2️⃣ 여기서 예측에 넘길 때만 0~1 스케일로 정규화
                lut256_for_pred = {
                    k: np.asarray(v, np.float32) / 4095.0 for k, v in {
                        "R_Low": lut256["R_Low"],
                        "G_Low": lut256["G_Low"],
                        "B_Low": lut256["B_Low"],
                        "R_High": high_R,
                        "G_High": high_G,
                        "B_High": high_B,
                    }.items()
                }

                # ✅ 예측은 정규화된 LUT 사용
                y_pred = self._predict_Y0W_from_models(
                    lut256_for_pred,
                    panel_text=panel, frame_rate=fr, model_year=model_year
                )

                # (선택) 디버그용 CSV 저장
                self._debug_dump_predicted_Y0W(
                    y_pred, tag=f"iter{it}_{panel}_fr{int(fr)}_my{int(model_year)%100:02d}", save_csv=True
                )

                # 이후 부분은 기존 그대로 유지
                d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)
                A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
                b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)
                mask  = np.isfinite(b_cat)
                A_use = A_cat[mask,:]; b_use = b_cat[mask]
                ATA = A_use.T @ A_use
                rhs = A_use.T @ b_use
                ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
                delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

                dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
                corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

                # ✅ 보정은 12bit 스케일에서 수행
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
            for k in new_lut_4096:
                new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

            # 7) UI 바로 업데이트 (차트+테이블)
            lut_plot = {
                "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
                "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
                "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
            }
            time.sleep(5)
            self._update_lut_chart_and_table(lut_plot)

            # 8) JSON 텍스트로 재조립 (TV write용)
            vac_json_optimized = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=new_lut_4096)

            # 9) 로딩 GIF 정지/완료 아이콘
            self._step_done(2)

            return vac_json_optimized, new_lut_4096

        except Exception as e:
            logging.exception("[PredictOpt] failed")
            # 로딩 애니 정리
            try:
                self._step_done(2)
            except Exception:
                pass
            return None, None
        
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

        logging.debug(f"[UI META] panel='{panel_text}', fr='{fr_val}Hz', model_year='Y{int(my_val):02d}'")
        return panel_text, fr_val, my_val
    
    # ===== [ADD] 정규화 다운샘플 & 업샘플 =====
    def _down4096_to_256_norm(self, arr4096):
        """4096 → 256 다운샘플 + [0,1] 정규화 (학습 스케일과 일치)"""
        a = np.asarray(arr4096, dtype=np.float32)
        idx = np.round(np.linspace(0, 4095, 256)).astype(int)
        return (a[idx] / 4095.0).astype(np.float32)

    def _up256_to_4096_norm(self, arr256_norm):
        """[0,1] 256 → [0,1] 4096 업샘플 (TV 적용 전 마지막에만 12bit 변환)"""
        arr256_norm = np.asarray(arr256_norm, dtype=np.float32)
        x_small = np.linspace(0, 1, 256)
        x_big   = np.linspace(0, 1, 4096)
        return np.interp(x_big, x_small, arr256_norm).astype(np.float32)

    def _to_tv_12bit(self, arr4096_norm):
        """[0,1] 4096 → 12bit 정수"""
        a = np.asarray(arr4096_norm, np.float32)
        return np.clip(np.round(a * 4095.0), 0, 4095).astype(int)

    def _build_runtime_X_from_db_json(self, vac_data_json: str):
        vac_dict = json.loads(vac_data_json)

        # 4096→256 정규화 (학습 스케일과 동일)
        lut256_norm = {
            "R_Low":  self._down4096_to_256_norm(vac_dict["RchannelLow"]),
            "R_High": self._down4096_to_256_norm(vac_dict["RchannelHigh"]),
            "G_Low":  self._down4096_to_256_norm(vac_dict["GchannelLow"]),
            "G_High": self._down4096_to_256_norm(vac_dict["GchannelHigh"]),
            "B_Low":  self._down4096_to_256_norm(vac_dict["BchannelLow"]),
            "B_High": self._down4096_to_256_norm(vac_dict["BchannelHigh"]),
        }

        # UI 메타 (model_year는 두 자리로 강제)
        panel_text, frame_rate, model_year_full = self._get_ui_meta()
        model_year_2digit = float(int(model_year_full) % 100)

        X_rows = [
            self._build_runtime_feature_row_W(
                lut256_norm, g,
                panel_text=panel_text,
                frame_rate=frame_rate,
                model_year_2digit=model_year_2digit
            )
            for g in range(256)
        ]
        X = np.vstack(X_rows).astype(np.float32)
        ctx = {"panel_text": panel_text, "frame_rate": frame_rate, "model_year_2digit": model_year_2digit}
        return X, lut256_norm, ctx
    
    def _debug_dump_predicted_Y0W(self, y_pred: dict, *, tag: str = "", save_csv: bool = True):
        """
        예측된 'W' 패턴 256포인트 (Gamma, Cx, Cy)를 로그로 요약 + (옵션) CSV 저장

        Parameters
        ----------
        y_pred : {"Gamma": (256,), "Cx": (256,), "Cy": (256,)}
        tag    : 로그/파일명 식별용 태그 (예: "iter1_INX_60_Y26")
        save_csv : True면 임시 CSV 파일로 저장 후 경로 로깅
        """
        import numpy as np, pandas as pd, tempfile, os, logging

        # 안전 가드
        req_keys = ("Gamma", "Cx", "Cy")
        if not all(k in y_pred for k in req_keys):
            logging.warning(f"[Predict/Debug] y_pred keys invalid: {list(y_pred.keys())}")
            return

        g = np.asarray(y_pred["Gamma"], dtype=np.float32)
        cx= np.asarray(y_pred["Cx"],    dtype=np.float32)
        cy= np.asarray(y_pred["Cy"],    dtype=np.float32)

        # ── 1) 통계 요약 로그
        def _stat(a, name):
            with np.errstate(invalid="ignore"):
                logging.debug(f"[Predict/Debug] {name}: "
                            f"shape={a.shape}, mean={np.nanmean(a):.6g}, std={np.nanstd(a):.6g}, "
                            f"min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}")
        _stat(g, "Gamma")
        _stat(cx,"Cx")
        _stat(cy,"Cy")

        # ── 2) 특정 인덱스 원소 출력 (0,1,2,127,128,129,254,255)
        idx_probe = [0,1,2,127,128,129,254,255]
        for i in idx_probe:
            if 0 <= i < 256:
                logging.debug(f"[Predict/Debug] g={i:3d} | Gamma={g[i]!r:>12} | Cx={cx[i]:.6f} | Cy={cy[i]:.6f}")

        # ── 3) (옵션) CSV 저장
        if save_csv:
            df = pd.DataFrame({"Gamma": g, "Cx": cx, "Cy": cy})
            safe_tag = "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in str(tag))
            with tempfile.NamedTemporaryFile(prefix=f"y0W_{safe_tag}_", suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
                df.to_csv(f.name, index_label="Gray")
                csv_path = f.name
            logging.info(f"[Predict/Debug] Y0(W) 256pts saved → {csv_path}")
    
    # ===== [ADD] 런타임 X 디버그 로깅 =====
    def _debug_log_runtime_X(self, X: np.ndarray, ctx: dict, tag="[RUNTIME X]"):
        # 기대: X.shape=(256,18)
        try:
            D = X.shape[1]
        except Exception:
            D = None
        logging.debug(f"{tag} shape={getattr(X,'shape',None)}, dim={D}")
        if X is None or X.shape != (256, 18):
            logging.warning(f"{tag} 스키마 불일치: 기대 (256,18), 실제 {getattr(X,'shape',None)}")

        # 컬럼 해석을 위해 인덱스 슬라이스
        idx = {
            "LUT": slice(0,6),
            "panel_onehot": slice(6,11),
            "fr": 11,
            "my": 12,
            "gray_norm": 13,
            "p_oh": slice(14,18),
        }

        # 패널 원핫 합/원핫성
        p_sum = X[:, idx["panel_onehot"]].sum(axis=1)
        uniq = np.unique(p_sum)
        logging.debug(f"{tag} panel_onehot sum unique: {uniq[:8]} (expect 0 or 1)")
        logging.debug(f"{tag} ctx: panel='{ctx.get('panel_text')}', fr={ctx.get('frame_rate')}, my(2digit)={ctx.get('model_year_2digit')}")

        # 샘플 행 (0, 128, -1) & tail12
        def _fmt_row(i):
            r = X[i]
            lut = ", ".join(f"{v:.4f}" for v in r[idx["LUT"]])
            tail = ", ".join(f"{v:.4f}" for v in r[-12:])
            return f"idx={i:3d} | LUT6=[{lut}] | tail12=[{tail}]"
        logging.debug(f"{tag} sample: {_fmt_row(0)}")
        logging.debug(f"{tag} sample: {_fmt_row(128)}")
        logging.debug(f"{tag} sample: {_fmt_row(255)}")

        # 마지막 10개 행의 tail & 회귀 타깃이 없으니 gray_norm만 체크
        for i in range(246, 256):
            r = X[i]
            tail12 = tuple(float(x) for x in r[-12:])
            logging.debug(f"{tag} last10 idx={i:3d} | gray_norm={r[idx['gray_norm']]:.4f} | tail12={tail12}")

    def _step_start(self, idx: int):
        """idx=1..5: 단계 시작(GIF on)"""
        label_widget = getattr(self.ui, f"vac_label_pixmap_step_{idx}")
        label, movie = self.start_loading_animation(label_widget, 'processing.gif')
        setattr(self, f"label_processing_step_{idx}", label)
        setattr(self, f"movie_processing_step_{idx}", movie)

    def _step_done(self, idx: int):
        """idx=1..5: 단계 종료(GIF off + 완료아이콘)"""
        label = getattr(self, f"label_processing_step_{idx}", None)
        movie = getattr(self, f"movie_processing_step_{idx}", None)
        if label is not None and movie is not None:
            self.stop_loading_animation(label, movie)
        # 완료 아이콘
        getattr(self.ui, f"vac_label_pixmap_step_{idx}").setPixmap(self.process_complete_pixmap)

    def _set_icon_scaled(self, label, pixmap):
        """라벨 현재 크기에 맞춰 아이콘 스케일 후 세팅"""
        if not pixmap or pixmap.isNull():
            label.clear(); return
        target_size = label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            # 레이아웃 직후 1프레임 뒤로 미루고 스케일 (안전장치)
            from PySide2.QtCore import QTimer
            QTimer.singleShot(0, lambda: self._set_icon_scaled(label, pixmap))
            return
        scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        
    def _set_icon_scaled(self, label, pixmap: QPixmap):
        if not label or pixmap is None or pixmap.isNull():
            return
        size = label.size()
        if size.width() <= 0 or size.height() <= 0:
            # 라벨이 아직 레이아웃되기 전이면 다음 프레임에 재시도
            QTimer.singleShot(0, lambda: self._set_icon_scaled(label, pixmap))
            return
        scaled = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    # ===== 내부용: step→라벨 위젯 찾기 =====
    def _step_label(self, step: int):
        # UI에 있는 라벨 이름 규칙: vac_label_pixmap_step_{n}
        return getattr(self.ui, f"vac_label_pixmap_step_{step}", None)

    # ===== 내부용: 진행중 애니메이션 핸들(라벨, 무비) 보관 =====
    def _ensure_step_anim_map(self):
        if not hasattr(self, "_step_anim"):
            self._step_anim = {}  # {step: (label, movie)}

    # ===== 공개 API: Step 시작/완료/실패 =====
    def _step_start(self, step: int):
        """해당 단계의 '처리중 GIF' 시작"""
        self._ensure_step_anim_map()
        lbl = self._step_label(step)
        if lbl is None:
            return
        # 이미 돌아가는 중이면 무시
        if step in self._step_anim:
            return
        label_handle, movie_handle = self.start_loading_animation(lbl, 'processing.gif')
        self._step_anim[step] = (label_handle, movie_handle)

    def _step_done(self, step: int):
        """해당 단계 애니 정지 + 완료 아이콘(스케일)"""
        self._ensure_step_anim_map()
        lbl = self._step_label(step)
        if lbl is None:
            return
        # 애니 정지
        if step in self._step_anim:
            try:
                label_handle, movie_handle = self._step_anim.pop(step)
                self.stop_loading_animation(label_handle, movie_handle)
            except Exception:
                pass
        # 완료 아이콘(라벨 크기에 맞춰)
        self._set_icon_scaled(lbl, self.process_complete_pixmap)

    def _step_fail(self, step: int):
        """해당 단계 애니 정지 + 실패 아이콘(스케일)"""
        self._ensure_step_anim_map()
        lbl = self._step_label(step)
        if lbl is None:
            return
        # 애니 정지
        if step in self._step_anim:
            try:
                label_handle, movie_handle = self._step_anim.pop(step)
                self.stop_loading_animation(label_handle, movie_handle)
            except Exception:
                pass
        # 실패 아이콘(라벨 크기에 맞춰)
        self._set_icon_scaled(lbl, self.process_fail_pixmap)

    def _step_set_pending(self, step: int):
        """대기(보류) 아이콘으로 교체"""
        lbl = self._step_label(step)
        if lbl is None:
            return
        self._set_icon_scaled(lbl, self.process_pending_pixmap)
    
    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        1. TV setting > VAC OFF → 측정 + UI 업데이트

        2. TV setting > VAC ON → DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 writing → 측정 + UI 업데이트

        3. 측정 결과 평가: 스펙 OK면 종료, NG면 자코비안 행렬을 통해 LUT 보정
        
        4. 보정 LUT TV에 Writing → 측정 + UI 업데이트

        5. 측정 결과 평가: 스펙 OK면 종료, NG면 자코비안 행렬을 통해 LUT 보정
        
        스펙 OK가 나올때까지 LUT 보정 반복...
        """
        base = cf.get_normalized_path(__file__, '..', '..', 'resources/images/pictures')
        self.process_complete_pixmap = QPixmap(os.path.join(base, 'process_complete.png'))
        self.process_fail_pixmap     = QPixmap(os.path.join(base, 'process_fail.png'))
        self.process_pending_pixmap  = QPixmap(os.path.join(base, 'process_pending.png'))
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        try:
            # 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            self.A_Gamma = self._build_A_from_artifacts(artifacts, "dGamma")   # (256, 6K)
            self.A_Cx    = self._build_A_from_artifacts(artifacts, "dCx")
            self.A_Cy    = self._build_A_from_artifacts(artifacts, "dCy")
            
            # # 예측 모델 로드
            # self.models_Y0_bundle = self._load_prediction_models()

        except FileNotFoundError as e:
            logging.error(f"[VAC Optimization] Jacobian file not found: {e}")

        except KeyError as e:
            logging.error(f"[VAC Optimization] Missing key in artifacts: {e}")

        except Exception as e:
            logging.exception("[VAC Optimization] Unexpected error occurred")
        
        # 1.2 TV VAC OFF 하기
        logging.info("[TV CONTROL] TV VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV CONTROL] TV VAC OFF 전환 성공")    
        # 1.3 OFF 측정 세션 시작
        logging.info("[MES] VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()
    #│                                                                                              │
    #└──────────────────────────────────────────────────────────────────────────────────────────────┘
    #################################################################################################
