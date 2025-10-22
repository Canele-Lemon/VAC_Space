        self._vac_dict_cache = None
        
        self.vac_optimization_gamma_chart = GammaChart(self.ui.vac_chart_gamma_3)
        self.vac_optimization_cie1976_chart = CIE1976Chart(self.ui.vac_chart_colorShift_2)
        self.vac_optimization_lut_chart = LUTChart(target_widget=self.ui.vac_graph_rgbLUT_4)

        self.vac_optimization_chromaticity_chart = XYChart(
            target_widget=self.ui.vac_chart_chromaticityDiff,
            x_label='Gray Level',
            y_label='Cx/Cy',
            x_range=(0, 256),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title=None,
            title_color='#595959',
            legend=False
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
            x_labels=self.colorshift_x_labels,
            y_label='delta u`v`',
            spec_line=0.04
        )

        self.expanded = True
        self.expand_VAC_Optimization_Result()
        self.ui.vac_btn_JSONdownload.setEnabled(False)
        
        ###########################################################################################    
    
    def _load_jacobian_artifacts(self):
        """
        jacobian_Y0_high.pkl 파일을 불러와서 artifacts 딕셔너리로 반환
        """
        jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian_Y0_high_INX_60_K33.pkl')
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
        저장된 자코비안 pkl로부터 A 행렬 (ΔY ≈ A·ΔH) 복원
        """
        # def stack_basis_all_grays(knots: np.ndarray, L=256) -> np.ndarray:
        #     """
        #     모든 그레이(0..255)에 대한 φ(g) K차원 가중치 행렬 (L x K)
        #     """
        #     def linear_interp_weights(g: int, knots: np.ndarray) -> np.ndarray:
        #         """
        #         그레이 g(0..255)에 대해, K개 knot에 대한 선형보간 '모자(hat)' 가중치 벡터 φ(g) 반환.
        #         - 양 끝은 1개, 중간은 2개 노드만 비영(희소)
        #         """
        #         K = len(knots)
        #         w = np.zeros(K, dtype=np.float32)
        #         # 왼쪽/오른쪽 경계
        #         if g <= knots[0]:
        #             w[0] = 1.0
        #             return w
        #         if g >= knots[-1]:
        #             w[-1] = 1.0
        #             return w
        #         # 내부: 인접한 두 knot 사이
        #         i = np.searchsorted(knots, g) - 1
        #         g0, g1 = knots[i], knots[i+1]
        #         t = (g - g0) / max(1, (g1 - g0))
        #         w[i]   = 1.0 - t
        #         w[i+1] = t
        #         return w
            
            # rows = [linear_interp_weights(g, knots) for g in range(L)]
            # return np.vstack(rows).astype(np.float32)

        knots = np.asarray(artifacts["knots"], dtype=np.int32)
        comp_obj = artifacts["components"][comp]
        coef = np.asarray(comp_obj["coef"], dtype=np.float32)
        scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)

        s = comp_obj["feature_slices"]
        s_high_R = slice(s["high_R"][0], s["high_R"][1])
        s_high_G = slice(s["high_G"][0], s["high_G"][1])
        s_high_B = slice(s["high_B"][0], s["high_B"][1])

        beta_R = coef[s_high_R] / np.maximum(scale[s_high_R], 1e-12)
        beta_G = coef[s_high_G] / np.maximum(scale[s_high_G], 1e-12)
        beta_B = coef[s_high_B] / np.maximum(scale[s_high_B], 1e-12)

        Phi = self._stack_basis(knots, L=256)

        A_R = Phi * beta_R.reshape(1, -1)
        A_G = Phi * beta_G.reshape(1, -1)
        A_B = Phi * beta_B.reshape(1, -1)

        A = np.hstack([A_R, A_G, A_B]).astype(np.float32)
        logging.info(f"[Jacobian] {comp} A 행렬 shape: {A.shape}") # (256, 3K)
        return A
    
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
            logging.debug(f"VAC ON 측정 결과:\n{self._off_store}")
            self.stop_loading_animation(self.label_processing_step_1, self.movie_processing_step_1)
            self.ui.vac_label_pixmap_step_1.setPixmap(self.process_complete_pixmap)
            logging.info("[STEP 1] VAC OFF 측정 - VAC OFF 상태 측정 완료")
            
            logging.info("[STEP 2] VAC 최적화 - VAC ON")
            if not self._set_vac_active(True):
                logging.warning("VAC ON 전환 실패 - VAC 최적화 종료")
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
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 종료")
            return

        # TV 쓰기 완료 시 콜백
        def _after_write(ok, msg):
            logging.info(f"[STEP 2] VAC 최적화 - 1차 최적화 LUT TV Writing 완료 : {msg}")
            if not ok:
                logging.error("VAC Writing 실패 - 종료")
                return
            # 쓰기 성공 → TV에서 VAC 읽어오기
            logging.info("[STEP 2] VAC 최적화 - 1차 최적화 LUT TV Reading 시작")
            self._read_vac_from_tv(_after_read)

        # TV에서 읽기 완료 시 콜백
        def _after_read(vac_dict):
            logging.info("[STEP 2] VAC 최적화 - 1차 최적화 LUT TV Reading 완료")
            if not vac_dict:
                logging.error("VAC 데이터 읽기 실패 - 종료")
                return

            # 캐시 보관 (TV 원 키명 유지)
            self._vac_dict_cache = vac_dict

            # LUT 차트는 "받을 때마다 전체 리셋 후 재그림"
            # TV 키명을 표준 표시용으로 바꿔서 전달 (RchannelHigh -> R_High 등)
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
                if self._check_spec_pass(self._off_store, self._on_store):
                    logging.info("✅ 스펙 통과 — 종료")
                    return
                # (D) 반복 보정 시작 (1회차)
                self._run_correction_iteration(iter_idx=1)

            # ── ON 측정 세션 시작 ──
            logging.info("[STEP 4] VAC 최적화 검증 - 1차 최적화 LUT 측정 시작")
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000, cs_settle_ms=1000,
                on_done=_after_on
            )

        # 3-b) VAC_Data TV에 writing
        logging.info("[STEP 2] VAC 최적화 - 1차 최적화 LUT TV Writing 시작")
        self._write_vac_to_tv(vac_data, on_finished=_after_write)
        
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
            
            # 1) 캐시/차트 갱신
            self._vac_dict_cache = vac_dict_after
            lut_dict_plot = {k.replace("channel","_"): v
                            for k, v in vac_dict_after.items() if "channel" in k}
            self._update_lut_chart_and_table(lut_dict_plot)
            
            # 2) ON 시리즈 리셋 (OFF는 참조 유지)
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()
            
            # 3) 보정 후(=ON) 측정 세션 시작
            profile_corr = SessionProfile(
                legend_text=f"CORR #{iter_idx}",   # state 판정은 'VAC OFF' prefix 여부로 하므로 여기선 ON 상태로 처리됨
                cie_label=None,                    # data_1/2 안 씀
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store          # 항상 OFF 대비 Δ를 계산
            )
            
            def _after_corr(store_corr):
                self._on_store = store_corr  # 최신 ON(보정 후) 측정 결과로 교체
                if self._check_spec_pass(self._off_store, self._on_store):
                    logging.info("✅ 스펙 통과 — 최적화 종료")
                    return
                if iter_idx < max_iters:
                    logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
                    self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
                else:
                    logging.info("⛔ 최대 보정 횟수 도달 — 종료")

            logging.info(f"보정 LUT 기준 측정 시작")
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

        # (아래 white/main 테이블 채우는 로직은 기존 그대로 유지)
        if pattern == 'white':
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
                # sub도 동일하게 적용
                ref_sub = profile.ref_store['gamma']['sub']['white'].get(gray, None)
                if ref_sub is not None:
                    _, cx_r_s, cy_r_s = ref_sub
                    if np.isfinite(cx_s):
                        d_cx_s = cx_s - cx_r_s
                        self._set_item_with_spec(
                            table_inst2, gray, cols['d_cx'], f"{d_cx_s:.6f}",
                            is_spec_ok=(abs(d_cx_s) <= 0.003)
                        )
                    if np.isfinite(cy_s):
                        d_cy_s = cy_s - cy_r_s
                        self._set_item_with_spec(
                            table_inst2, gray, cols['d_cy'], f"{d_cy_s:.6f}",
                            is_spec_ok=(abs(d_cy_s) <= 0.003)
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
        'main': (x, y, lv, cct, duv)  또는  None,
        'sub' : (x, y, lv, cct, duv)  또는  None
        }
        """
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        # 현재 세션 상태
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                store['colorshift'][role].append((np.nan, np.nan, np.nan, np.nan))
                continue

            x, y, lv, cct, duv = res

            # xy → u′v′ 변환
            u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))
            store['colorshift'][role].append((float(x), float(y), float(u_p), float(v_p)))

            # ▶▶ 차트 업데이트 (간소화 API)
            # CIE1976ChartVAC: add_point(state, role, u_p, v_p)
            self.vac_optimization_cie1976_chart.add_point(
                state=state,
                role=role,        # 'main' 또는 'sub'
                u_p=float(u_p),
                v_p=float(v_p)
            )

    def _finalize_session(self):
        s = self._sess
        profile: SessionProfile = s['profile']
        table = self.ui.vac_table_opt_mes_results_main
        cols = profile.table_cols
        thr_gamma = 0.05

        # white/main 감마 계산
        lv_series = np.zeros(256, dtype=np.float64)
        for g in range(256):
            tup = s['store']['gamma']['main']['white'].get(g, None)
            lv_series[g] = float(tup[0]) if tup else np.nan
        gamma_vec = self._compute_gamma_series(lv_series)
        for g in range(256):
            if np.isfinite(gamma_vec[g]):
                self._set_item(table, g, cols['gamma'], f"{gamma_vec[g]:.6f}")

        # ΔGamma (ON/보정 시)
        if profile.ref_store is not None and 'd_gamma' in cols:
            ref_lv = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = profile.ref_store['gamma']['main']['white'].get(g, None)
                ref_lv[g] = float(tup[0]) if tup else np.nan
            ref_gamma = self._compute_gamma_series(ref_lv)
            dG = gamma_vec - ref_gamma
            for g in range(256):
                if np.isfinite(dG[g]):
                    self._set_item_with_spec(
                        table, g, cols['d_gamma'], f"{dG[g]:.6f}",
                        is_spec_ok=(abs(dG[g]) <= thr_gamma)
                    )

        if callable(s['on_done']):
            try:
                s['on_done'](s['store'])
            except Exception as e:
                logging.exception(e)
                
    def _ensure_row_count(self, table, row_idx):
        if table.rowCount() <= row_idx:
            table.setRowCount(row_idx + 1)

    def _set_item(self, table, row, col, value):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))

        table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
        
    def _set_item_with_spec(self, table, row, col, value, *, is_spec_ok: bool):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))
        # 🔸 스펙 OUT만 빨간 배경
        if is_spec_ok:
            item.setBackground(QColor(255, 255, 255))  # 기본(흰색)로 돌림
        else:
            item.setBackground(QColor(255, 200, 200))  # 연한 빨강

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
            
    def _focus_cell(self, table: QtWidgets.QTableWidget, row: int, col: int, center: bool=True):
        """해당 셀을 선택하고 스크롤로 가시화."""
        table.setCurrentCell(row, col)  # 선택 & 포커스
        item = table.item(row, col)
        if item is None:
            return
        hint = (QtWidgets.QAbstractItemView.PositionAtCenter
                if center else QtWidgets.QAbstractItemView.PositionAtBottom)
        table.scrollToItem(item, hint)

    def _flash_cell(self, table: QtWidgets.QTableWidget, row: int, col: int,
                    ms: int=350, color: QtGui.QColor=QtGui.QColor(255, 236, 179)):
        """셀 배경을 잠깐 하이라이트(시각적 피드백)."""
        item = table.item(row, col)
        if item is None:
            return
        old_brush = item.background()
        item.setBackground(QtGui.QBrush(color))
        QtCore.QTimer.singleShot(ms, lambda: item.setBackground(old_brush))

    def start_VAC_optimization(self):
        
        """
        =============== 메인 엔트리: 버튼 이벤트 연결용 ===============
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
            # 0. 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")   # (256, 3K)
            self.A_Cx    = self._build_A_from_artifacts(artifacts, "Cx")
            self.A_Cy    = self._build_A_from_artifacts(artifacts, "Cy")

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
        logging.info("[STEP 1] VAC OFF 측정 - VAC OFF")
        if not self._set_vac_active(False):
            logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
            
        # 1.3 OFF 측정 세션 시작
        logging.info("[STEP 1] VAC OFF 측정 - VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()
