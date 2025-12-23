학습이 다 끝났습니다.
먼저 만든 자코비안 J_g와 함께 예측 모델을 활용해서 아래 _generate_predicted_vac_lut 메서드를 작성하려고 합니다.

     def _apply_predicted_vac_and_measure_on(self):
        self._step_start(2)
        BASE_VAC_PK = 3025
        vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(BASE_VAC_PK)
        if base_vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        base_vac_dict = json.loads(base_vac_data)
        self._vac_dict_cache = base_vac_dict
        
        # 예측 모델을 통한 시야각 특성 예측 -> 자코비안 보정 들어가야 함
        ##########################################################################################
        # try:
        #     predicted_vac_data, debug_info = self._generate_predicted_vac_lut(
        #         base_vac_dict,
        #         n_iters=2,
        #         wG=0.4,
        #         wC=1.0,
        #         lambda_ridge=1e-3
        #     )
        # except Exception:
        #     logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
        #     predicted_vac_json, debug_info = None, None
        ##########################################################################################    
            
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in base_vac_dict.items() if "channel" in key}
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
            mismatch_keys = self._verify_vac_data_match(written_data=base_vac_dict, read_data=read_vac_dict)

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
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=1))
                self._spec_thread.start()

            logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                on_done=_after_on
            )

        logging.info("[VAC Writing] 예측기반 최적화 VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(base_vac_data, on_finished=_after_write)

현재까지 작성된 _generate_predicted_vac_lut는 다음과 같습니다. 어떻게 수정하면 되나요? 제가 원하는건 DB fetch한 base_vac_data 기준 시야각 특성 예측 -> 자코비안 보정입니다.
    def _generate_predicted_vac_lut(self, vac_dict, *, n_iters=1, wG=0.4, wC=1.0, lambda_ridge=1e-3):
        """
        JSON 로드한 Base VAC Data를 입력으로 받아,
        예측 모델(OFF 대비 dCx/dCy/dGamma 예측) + 자코비안 J_g 를 이용해
        High LUT(R_High/G_High/B_High)를 n_iters회 반복 보정한 후,
        4096포인트 LUT와 TV write용 VAC Data를 생성해서 return 합니다.

        Returns
        -------
        vac_json_optimized : str or None
            TV에 write할 VAC JSON (표준 포맷). 실패 시 None.
        new_lut_4096 : dict or None
            최종 4096포인트 LUT 딕셔너리. 실패 시 None.
        """
        try:
            self._load_mapping_index_gray_to_lut()
            
            idx_map = np.asarray(self._mapping_index_gray_to_lut, dtype=np.int32)
            idx_map_f = idx_map.astype(np.float32)
            print("idx_map:")
            print(idx_map)
            print("idx_map_f:")
            print(idx_map_f)
            
            R_low_4096 = np.asarray(vac_dict["RchannelLow"],  dtype=np.float32)
            G_low_4096 = np.asarray(vac_dict["GchannelLow"],  dtype=np.float32)
            B_low_4096 = np.asarray(vac_dict["BchannelLow"],  dtype=np.float32)
            R_high_4096 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32)
            G_high_4096 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32)
            B_high_4096 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32)
            
            lut256 = {
                "R_Low":  R_low_4096[idx_map],
                "R_High": R_high_4096[idx_map],
                "G_Low":  G_low_4096[idx_map],
                "G_High": G_high_4096[idx_map],
                "B_Low":  B_low_4096[idx_map],
                "B_High": B_high_4096[idx_map],
            }

            if not hasattr(self, "_J_dense") or self._J_dense is None:
                logging.error("[PredictOpt] J_g bundle (_J_dense) not prepared.")
                return None, None

            panel, fr, model_year = self._get_ui_meta()

            high_R_ctrl = lut256["R_High"].copy()
            high_G_ctrl = lut256["G_High"].copy()
            high_B_ctrl = lut256["B_High"].copy()

            for it in range(1, n_iters + 1):
                d_lut256_for_pred = {
                    "dR_Low": np.abs(lut256["R_Low"] - idx_map_f),
                    "dG_Low": np.abs(lut256["G_Low"] - idx_map_f),
                    "dB_Low": np.abs(lut256["B_Low"] - idx_map_f),
                    "dR_High": np.abs(high_R_ctrl - idx_map_f),
                    "dG_High": np.abs(high_G_ctrl - idx_map_f),
                    "dB_High": np.abs(high_B_ctrl - idx_map_f),
                }

# #######################################################################################3
#                 y_pred = self._predict_Y_from_models(
#                     d_lut256_for_pred,
#                     panel_text=panel,
#                     frame_rate=fr,
#                     model_year=model_year
#                 )

#                 # 이후 부분은 기존 그대로 유지
#                 d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)
#                 A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
#                 b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)
#                 mask  = np.isfinite(b_cat)
#                 A_use = A_cat[mask,:]; b_use = b_cat[mask]
#                 ATA = A_use.T @ A_use
#                 rhs = A_use.T @ b_use
#                 ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
#                 delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

#                 dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
#                 corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

#                 # ✅ 보정은 12bit 스케일에서 수행
#                 high_R = np.clip(self._enforce_monotone(high_R + corr_R), 0, 4095)
#                 high_G = np.clip(self._enforce_monotone(high_G + corr_G), 0, 4095)
#                 high_B = np.clip(self._enforce_monotone(high_B + corr_B), 0, 4095)

#                 logging.info(f"[PredictOpt] iter {it} done. (wG={wG}, wC={wC})")

#             # 6) 256→4096 업샘플 (Low는 그대로, High만 갱신)
#             new_lut_4096 = {
#                 "RchannelLow":  np.asarray(vac_dict["RchannelLow"], dtype=np.float32),
#                 "GchannelLow":  np.asarray(vac_dict["GchannelLow"], dtype=np.float32),
#                 "BchannelLow":  np.asarray(vac_dict["BchannelLow"], dtype=np.float32),
#                 "RchannelHigh": self._up256_to_4096(high_R),
#                 "GchannelHigh": self._up256_to_4096(high_G),
#                 "BchannelHigh": self._up256_to_4096(high_B),
#             }
#             for k in new_lut_4096:
#                 new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

#             # 7) UI 바로 업데이트 (차트+테이블)
#             lut_plot = {
#                 "R_Low":  new_lut_4096["RchannelLow"],  "R_High": new_lut_4096["RchannelHigh"],
#                 "G_Low":  new_lut_4096["GchannelLow"],  "G_High": new_lut_4096["GchannelHigh"],
#                 "B_Low":  new_lut_4096["BchannelLow"],  "B_High": new_lut_4096["BchannelHigh"],
#             }
#             time.sleep(5)
#             self._update_lut_chart_and_table(lut_plot)

#             # 8) JSON 텍스트로 재조립 (TV write용)
#             vac_json_optimized = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=new_lut_4096)

#             # 9) 로딩 GIF 정지/완료 아이콘
#             self._step_done(2)

#             return vac_json_optimized, new_lut_4096

        except Exception as e:
            logging.exception("[PredictOpt] failed")
            # 로딩 애니 정리
            try:
                self._step_done(2)
            except Exception:
                pass
            return None, None
