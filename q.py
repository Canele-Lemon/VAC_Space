    def _run_batch_correction_with_jacobian(self, iter_idx, max_iters, thr_gamma, thr_c, lam=1e-3, metrics=None):
        logging.info(f"[Batch Correction] iteration {iter_idx} start (Jacobian dense)")

        # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
        if not hasattr(self, "_J_dense"):
            logging.error("[Batch Correction] J_dense not loaded") # self._J_dense 없음
            return
        self._load_mapping_index_gray_to_lut()
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.error("[Batch Correction] no VAC cache; need latest TV VAC JSON")
            return

        # 1) NG gray 리스트 / Δ 타깃 준비
        if metrics is not None and "ng_grays" in metrics and "dG" in metrics:
            ng_list = list(metrics["ng_grays"])
            d_targets = {
                "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
                "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
                "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
            }
            thr_gamma = float(metrics.get("thr_gamma", thr_gamma))
            thr_c     = float(metrics.get("thr_c",     thr_c))
            logging.info(f"[Batch Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
        else:
            dG, dCx, dCy, ng_list = SpecEvalThread.compute_gray_errors_and_ng_list(
                self._off_store, self._on_store,
                thr_gamma=thr_gamma, thr_c=thr_c
            )
            d_targets = {
                "Gamma": dG.astype(np.float32),
                "Cx":    dCx.astype(np.float32),
                "Cy":    dCy.astype(np.float32),
            }
            logging.info(f"[Batch Correction] NG grays (recomputed): {ng_list}")

        if not ng_list:
            logging.info("[Batch Correction] no NG gray (또는 0/1/254/255만 NG) → 보정 없음")
            return
    
        # 2) 현재 High LUT 확보
        vac_dict = self._vac_dict_cache

        RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
        GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
        BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

        RH = RH0.copy()
        GH = GH0.copy()
        BH = BH0.copy()

        # 3) index별 Δ 누적 (여러 gray가 같은 index를 참조할 수 있으므로)
        delta_acc = {
            "R": np.zeros_like(RH),
            "G": np.zeros_like(GH),
            "B": np.zeros_like(BH),
        }
        count_acc = {
            "R": np.zeros_like(RH, dtype=np.int32),
            "G": np.zeros_like(GH, dtype=np.int32),
            "B": np.zeros_like(BH, dtype=np.int32),
        }

        mapLUT = self._lut_map_high["R"] # (256,)
        
        n_gray = 256
        dR_gray = np.full(n_gray, np.nan, np.float32)
        dG_gray = np.full(n_gray, np.nan, np.float32)
        dB_gray = np.full(n_gray, np.nan, np.float32)
        corr_flag = np.zeros(n_gray, np.int32)
        
        # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
        for g in ng_list:
            Jg = self._J_dense[g]
            
            if 0 <= g < n_gray:
                corr_flag[g] = 1
                
            dX = self._solve_delta_rgb_for_gray(
                g,
                d_targets,
                lam=lam,
                thr_c=thr_c,          # 색좌표 스펙 (예: 0.003)
                thr_gamma=thr_gamma,  # 감마 스펙 (예: 0.05)
                base_wCx=0.5,         # Cx 기본 가중치 (기존 0.5를 base로 사용)
                base_wCy=0.5,         # Cy 기본 가중치
                base_wG=1.0,          # Gamma 기본 가중치
                boost=3.0,            # NG일 때 배율
                keep=0.2,             # OK일 때 배율 (거의 무시)
            )
            if dX is None:
                continue

            dR, dG, dB = dX
            
            if 0 <= g < n_gray:
                dR_gray[g] = dR
                dG_gray[g] = dG
                dB_gray[g] = dB

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
        for ch, arr, arr0 in (
            ("R", RH, RH0),
            ("G", GH, GH0),
            ("B", BH, BH0),
        ):
            da = delta_acc[ch]
            ct = count_acc[ch]
            mask = ct > 0

            if not np.any(mask):
                logging.info(f"[Batch Correction] channel {ch}: no indices updated")
                continue

            # 평균 Δ
            arr[mask] = arr0[mask] + (da[mask] / ct[mask])
            # clip
            arr[:] = np.clip(arr, 0.0, 4095.0)
            # 단조 증가 (i<j → LUT[i] ≤ LUT[j])
            self._enforce_monotone(arr)

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
            
        df_corr = self._build_batch_corr_df(
                iter_idx=iter_idx,
                d_targets=d_targets,
                dR_gray=dR_gray,
                dG_gray=dG_gray,
                dB_gray=dB_gray,
                corr_flag=corr_flag,
                mapR=mapR, mapG=mapG, mapB=mapB,
                RH0=RH0, GH0=GH0, BH0=BH0,
                RH=RH, GH=GH, BH=BH
            )
        logging.info(
            f"[Batch Correction] {iter_idx}회차 보정 결과:\n"
            + df_corr.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )
        self._save_batch_corr_df(iter_idx, df_corr)
                   
        # 보정 LUT 시각화
        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],
            "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],
            "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],
            "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)

        # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
        logging.info(f"[VAC Writing] LUT {iter_idx}차 보정 VAC Data TV Writing start")

        vac_write_json = self.build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )
        vac_dict = json.loads(vac_write_json)
        self._vac_dict_cache = vac_dict

        def _after_write(ok, msg):
            logging.info(f"[VAC Writing] write result: {ok} {msg}")
            if not ok:
                return
            logging.info("[VAC Reading] TV reading after write")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'restart panelcontroller')
            time.sleep(1.0)
            self.send_command(self.ser_tv, 'exit')
            if not vac_dict_after:
                logging.error("[VAC Reading] TV read-back failed")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=vac_dict, read_data=vac_dict_after)
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
                legend_text=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7,
                            "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr
                self._update_last_on_lv_norm(store_corr)
                
                self._step_start(5)
                self._spec_thread = SpecEvalThread(
                    self._off_store, self._on_store,
                    thr_gamma=thr_gamma, thr_c=thr_c, parent=self
                )
                self._spec_thread.finished.connect(
                    lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx, max_iters)
                )
                self._spec_thread.start()

            logging.info(f"[Measurement] LUT {iter_idx}차 보정 기준 re-measure start (after LUT update)")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_corr,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                gamma_settle_ms=1000,
                cs_settle_ms=1000,
                on_done=_after_corr
            )

        self._step_start(3)
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)

    def _build_batch_corr_df(
        self,
        iter_idx: int,
        d_targets: dict,
        dR_gray: np.ndarray,
        dG_gray: np.ndarray,
        dB_gray: np.ndarray,
        corr_flag: np.ndarray,
        mapR: np.ndarray,
        mapG: np.ndarray,
        mapB: np.ndarray,
        RH0: np.ndarray, GH0: np.ndarray, BH0: np.ndarray,
        RH:  np.ndarray, GH:  np.ndarray, BH:  np.ndarray,
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
            idxR = int(mapR[g]) if 0 <= g < len(mapR) else -1
            idxG = int(mapG[g]) if 0 <= g < len(mapG) else -1
            idxB = int(mapB[g]) if 0 <= g < len(mapB) else -1

            row = {
                "gray": int(g),
                "LUT idx": idxR,  # 기준으로 R High 인덱스를 사용
                "CORR": int(corr_flag[g]),  # 1: 이 gray는 이번 회차 보정 대상(NG), 0: OK
                "ΔCx": float(d_targets["Cx"][g]) if np.isfinite(d_targets["Cx"][g]) else np.nan,
                "ΔCy": float(d_targets["Cy"][g]) if np.isfinite(d_targets["Cy"][g]) else np.nan,
                "ΔGamma": float(d_targets["Gamma"][g]) if np.isfinite(d_targets["Gamma"][g]) else np.nan,
                "ΔR": float(dR_gray[g]) if np.isfinite(dR_gray[g]) else 0.0,
                "ΔG": float(dG_gray[g]) if np.isfinite(dG_gray[g]) else 0.0,
                "ΔB": float(dB_gray[g]) if np.isfinite(dB_gray[g]) else 0.0,
            }

            # R before/after
            if 0 <= idxR < len(RH0):
                row["R_before"] = float(RH0[idxR])
                row["R_after"]  = float(RH[idxR])
            else:
                row["R_before"] = np.nan
                row["R_after"]  = np.nan

            # G
            if 0 <= idxG < len(GH0):
                row["G_before"] = float(GH0[idxG])
                row["G_after"]  = float(GH[idxG])
            else:
                row["G_before"] = np.nan
                row["G_after"]  = np.nan

            # B
            if 0 <= idxB < len(BH0):
                row["B_before"] = float(BH0[idxB])
                row["B_after"]  = float(BH[idxB])
            else:
                row["B_before"] = np.nan
                row["B_after"]  = np.nan

            rows.append(row)

        df_corr = pd.DataFrame(rows, columns=[
            "gray", "LUT idx", "CORR",
            "ΔCx", "ΔCy", "ΔGamma",
            "ΔR", "ΔG", "ΔB",
            "R_before", "R_after",
            "G_before", "G_after",
            "B_before", "B_after",
        ])

        self._last_batch_corr_df = df_corr
        
        return df_corr

또 각 gray 마다 가중치도 df에 나타나도록 위 코드를 수정해주세요
