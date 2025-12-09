def _predictive_first_optimize(self, vac_data_json, *, n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3):
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

        # 이후 업샘플, UI 업데이트, JSON 빌드는 기존 코드 그대로