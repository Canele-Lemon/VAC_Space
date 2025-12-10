def _generate_predicted_vac_lut(
    self,
    vac_dict,
    *,
    n_iters: int = 2,
    wG: float = 0.4,
    wC: float = 1.0,
    lambda_ridge: float = 1e-3
):
    """
    Base VAC LUT(vac_dict)을 입력으로 받아,
    예측 모델(OFF 대비 dCx/dCy/dGamma 예측) + 자코비안 J_g 를 이용해
    High LUT(R/G/B)를 n_iters 회 반복 보정한 후,
    4096포인트 LUT와 TV write용 VAC JSON을 생성해서 반환한다.

    Returns
    -------
    vac_json_optimized : str or None
        TV에 write할 VAC JSON (표준 포맷). 실패 시 None.
    new_lut_4096 : dict or None
        최종 4096포인트 LUT 딕셔너리. 실패 시 None.
    """
    try:
        # ------------------------------------------------------------------
        # 1) 4096 → 256 다운샘플 (12bit 스케일 그대로)
        # ------------------------------------------------------------------
        lut256 = {
            "R_Low":  self._down4096_to_256_float(vac_dict["RchannelLow"]),
            "R_High": self._down4096_to_256_float(vac_dict["RchannelHigh"]),
            "G_Low":  self._down4096_to_256_float(vac_dict["GchannelLow"]),
            "G_High": self._down4096_to_256_float(vac_dict["GchannelHigh"]),
            "B_Low":  self._down4096_to_256_float(vac_dict["BchannelLow"]),
            "B_High": self._down4096_to_256_float(vac_dict["BchannelHigh"]),
        }

        # 자코비안 번들이 준비되어 있는지 확인
        if not hasattr(self, "_J_dense") or self._J_dense is None:
            logging.error("[PredictOpt] J_g bundle (_J_dense) not prepared.")
            return None, None

        # 패널 메타데이터 (예측 모델 입력용)
        panel, fr, model_year = self._get_ui_meta()

        # 보정 대상 High LUT (12bit 스케일, 256포인트)
        high_R = lut256["R_High"].copy()
        high_G = lut256["G_High"].copy()
        high_B = lut256["B_High"].copy()

        # ------------------------------------------------------------------
        # 2) 반복 보정 루프
        # ------------------------------------------------------------------
        for it in range(1, n_iters + 1):
            # 2-1) 예측에 사용할 LUT를 0~1 스케일로 정규화
            lut256_for_pred = {
                k: np.asarray(v, np.float32) / 4095.0
                for k, v in {
                    "R_Low":  lut256["R_Low"],
                    "G_Low":  lut256["G_Low"],
                    "B_Low":  lut256["B_Low"],
                    "R_High": high_R,
                    "G_High": high_G,
                    "B_High": high_B,
                }.items()
            }

            # 2-2) VAC OFF 대비 dCx/dCy/dGamma 예측
            #      (모델이 이미 ON-OFF 값을 학습했다고 가정)
            y_pred = self._predict_Y0W_from_models(
                lut256_for_pred,
                panel_text=panel,
                frame_rate=fr,
                model_year=model_year,
            )
            # 디버그용 CSV 덤프 (선택)
            self._debug_dump_predicted_Y0W(
                y_pred,
                tag=f"iter{it}_{panel}_fr{int(fr)}_my{int(model_year) % 100:02d}",
                save_csv=True,
            )

            # 2-3) Δtarget 벡터 구성
            #      예측 결과 자체가 OFF 대비 (dCx, dCy, dGamma)이므로 그대로 사용
            d_targets = {
                "Cx":    np.asarray(y_pred["Cx"],    dtype=np.float32),
                "Cy":    np.asarray(y_pred["Cy"],    dtype=np.float32),
                "Gamma": np.asarray(y_pred["Gamma"], dtype=np.float32),
            }

            # 2-4) gray별 ΔR/ΔG/ΔB 계산
            dR_vec = np.zeros(256, dtype=np.float32)
            dG_vec = np.zeros(256, dtype=np.float32)
            dB_vec = np.zeros(256, dtype=np.float32)

            # 스펙 기준값 (가중치 자동 계산용)
            thr_c = 0.003
            thr_gamma = 0.05

            for g in range(256):
                # 양 끝단(0,1,254,255)은 자코비안 신뢰도/감마 정의 문제로 스킵
                if g < 2 or g > 253:
                    continue

                res = self._solve_delta_rgb_for_gray(
                    g,
                    d_targets,
                    lam=lambda_ridge,
                    # NG 정도에 따라 자동 가중치 사용
                    thr_c=thr_c,
                    thr_gamma=thr_gamma,
                    base_wCx=wC,
                    base_wCy=wC,
                    base_wG=wG,
                    boost=3.0,
                    keep=0.2,
                )
                if res is None:
                    continue

                dR, dG, dB, wCx_eff, wCy_eff, wG_eff, step_gain = res
                dR_vec[g] = dR
                dG_vec[g] = dG
                dB_vec[g] = dB

            # 2-5) LUT 갱신 (12bit 스케일에서 적용 + 단조성 강제 + 클리핑)
            high_R = np.clip(self._enforce_monotone(high_R + dR_vec), 0, 4095)
            high_G = np.clip(self._enforce_monotone(high_G + dG_vec), 0, 4095)
            high_B = np.clip(self._enforce_monotone(high_B + dB_vec), 0, 4095)

            logging.info(
                f"[PredictOpt(Jg)] iter {it} done. "
                f"(wG={wG}, wC={wC}, λ={lambda_ridge})"
            )

        # ------------------------------------------------------------------
        # 3) 256 → 4096 업샘플 + 최종 LUT/JSON 생성
        # ------------------------------------------------------------------
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"], dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"], dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"], dtype=np.float32),
            "RchannelHigh": self._up256_to_4096(high_R),
            "GchannelHigh": self._up256_to_4096(high_G),
            "BchannelHigh": self._up256_to_4096(high_B),
        }

        # 정수 0~4095 범위로 정리
        for k in new_lut_4096:
            new_lut_4096[k] = np.clip(
                np.round(new_lut_4096[k]), 0, 4095
            ).astype(np.uint16)

        # GUI 그래프/테이블 업데이트용
        lut_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],
            "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],
            "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],
            "B_High": new_lut_4096["BchannelHigh"],
        }
        time.sleep(5)
        self._update_lut_chart_and_table(lut_plot)

        # TV write용 표준 VAC JSON 조립
        vac_json_optimized = self.build_vacparam_std_format(
            base_vac_dict=vac_dict,
            new_lut_tvkeys=new_lut_4096,
        )

        # Step2 완료 아이콘 업데이트
        try:
            self._step_done(2)
        except Exception:
            pass

        return vac_json_optimized, new_lut_4096

    except Exception:
        logging.exception("[PredictOpt] failed")
        # 실패해도 Step2 로딩 애니는 정리
        try:
            self._step_done(2)
        except Exception:
            pass
        return None, None