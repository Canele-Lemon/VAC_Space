def _predictive_first_optimize(self, vac_data_json, *, 
                               n_iters=2, wG=0.4, wC=1.0, lambda_ridge=1e-3,
                               optimize_focus='balanced',   # ← 추가
                               gamma_soft=0.0               # ← 'chroma_soft'일 때 >0로
                               ):
    """
    optimize_focus: 'balanced' | 'chroma' | 'chroma_soft'
      - balanced   : 기존과 동일
      - chroma     : Cx/Cy만 사용 (Gamma 완전 배제)
      - chroma_soft: Cx/Cy 위주 + ΔGamma≈0 소프트 제약 (gamma_soft로 강도 조절)
    """
    try:
        vac_dict = json.loads(vac_data_json)
        # ... (중략: 기존 코드 동일) ...

        for it in range(1, n_iters + 1):
            # (예측/디버그 동일)
            y_pred = self._predict_Y0W_from_models(
                lut256_for_pred,
                panel_text=panel, frame_rate=fr, model_year=model_year
            )
            self._debug_dump_predicted_Y0W(
                y_pred, tag=f"iter{it}_{panel}_fr{int(fr)}_my{int(model_year)%100:02d}", save_csv=True
            )

            # Δ 타깃
            d_targets = self._delta_targets_vs_OFF_from_pred(y_pred, self._off_store)

            # ====== ★ 선형계 구성: 모드별로 다르게 ======
            blocks = []
            rhs    = []
            if optimize_focus == 'balanced':
                if np.any(np.isfinite(d_targets["Gamma"])):
                    blocks.append(wG*self.A_Gamma)
                    rhs.append( -wG*d_targets["Gamma"] )
                blocks.append(wC*self.A_Cx); rhs.append( -wC*d_targets["Cx"] )
                blocks.append(wC*self.A_Cy); rhs.append( -wC*d_targets["Cy"] )

            elif optimize_focus == 'chroma':
                # 감마 완전히 제외
                blocks.append(wC*self.A_Cx); rhs.append( -wC*d_targets["Cx"] )
                blocks.append(wC*self.A_Cy); rhs.append( -wC*d_targets["Cy"] )

            elif optimize_focus == 'chroma_soft':
                # Cx/Cy는 정상, 감마는 '0 타깃' 소프트 제약
                blocks.append(wC*self.A_Cx); rhs.append( -wC*d_targets["Cx"] )
                blocks.append(wC*self.A_Cy); rhs.append( -wC*d_targets["Cy"] )
                if gamma_soft > 0.0:
                    blocks.append(gamma_soft * self.A_Gamma)
                    rhs.append( np.zeros(256, dtype=np.float32) )  # ΔGamma≈0 제약

            else:
                logging.warning(f"[PredictOpt] unknown optimize_focus={optimize_focus}, fallback balanced")
                blocks.append(wG*self.A_Gamma); rhs.append( -wG*d_targets["Gamma"] )
                blocks.append(wC*self.A_Cx);   rhs.append( -wC*d_targets["Cx"] )
                blocks.append(wC*self.A_Cy);   rhs.append( -wC*d_targets["Cy"] )

            A_cat = np.vstack([b.astype(np.float32) for b in blocks])
            b_cat = -np.concatenate([r.astype(np.float32) for r in rhs])

            mask  = np.isfinite(b_cat)
            A_use = A_cat[mask,:]; b_use = b_cat[mask]

            ATA = A_use.T @ A_use
            rhs = A_use.T @ b_use
            ATA[np.diag_indices_from(ATA)] += float(lambda_ridge)
            delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

            # 이후 곡선적용/클램프 동일 ...