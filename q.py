    def _build_features_for_gray_pred(self, X_dict, gray: int) -> np.ndarray:
        """
        예측모델(Y0) 학습용 feature row 구성.

        X_row(g) = [
            dR_Low, dG_Low, dB_Low,
            dR_High, dG_High, dB_High,
            panel_maker(one-hot...),
            frame_rate,
            model_year,
            gray_norm (= g/255),
            LUT_index_j (mapping_j[g])
        ]

        여기서 dR_Low 등은 VACInputBuilder.prepare_X_delta_lut_with_mapping()
        에서 bypass LUT 기준 ΔLUT로 계산된 값이라고 가정한다.
        """
        delta_lut = X_dict["lut_delta_raw"]   # dict: ch -> (256,) float32 (bypass 기준 ΔLUT)
        meta      = X_dict["meta"]            # dict: panel_maker(one-hot), frame_rate, model_year
        j_map     = X_dict["mapping_j"]       # (256,) int32

        row = []

        # 1) dLUT 6채널 (Low + High)
        for ch in ("R_Low", "G_Low", "B_Low", "R_High", "G_High", "B_High"):
            row.append(float(delta_lut[ch][gray]))

        # 2) panel_maker one-hot
        row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())

        # 3) frame_rate, model_year
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # 4) gray_norm
        row.append(gray / 255.0)

        # 5) LUT index (mapping)
        j_idx = int(j_map[gray])  # 0..4095
        row.append(float(j_idx))

        return np.asarray(row, dtype=np.float32)
        
        
        
    def build_XYdataset_for_prediction_Y0(self, pattern: str = "W"):
        """
        예측모델(Y0: dCx, dCy, dGamma) 학습용 X, Y0, groups를 생성한다.

        - X: bypass 기준 dR/G/B_Low/High + panel meta + gray_norm + LUT index
        - Y0: pattern(기본 'W') 기준 (target 측정 - bypass 측정) 값
              "dCx", "dCy", "dGamma" 3개 타깃을 한 번에 반환

        Returns
        -------
        X : np.ndarray (N, D)
        Y0: dict[str, np.ndarray]
            {
              "dCx":    (N,),
              "dCy":    (N,),
              "dGamma": (N,),
            }
        groups : np.ndarray (N,)
            각 행이 어느 PK에서 왔는지 나타내는 그룹 벡터
            (GroupKFold / GroupShuffleSplit 등에 사용)
        """
        X_rows = []
        y_dCx  = []
        y_dCy  = []
        y_dG   = []
        groups = []

        for s in self.samples:
            pk = s["pk"]
            Xd = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
            Yd = s["Y"]  # {"Y0": {...}, ...}

            if "Y0" not in Yd:
                logging.warning(f"[VACDataset] PK={pk}: Y dict에 'Y0'가 없습니다. 건너뜀.")
                continue
            if pattern not in Yd["Y0"]:
                logging.warning(f"[VACDataset] PK={pk}: Y0에 pattern='{pattern}'가 없습니다. 건너뜀.")
                continue

            y0 = Yd["Y0"][pattern]   # {"dCx":(256,), "dCy":(256,), "dGamma":(256,)}

            vec_cx = np.asarray(y0["dCx"],    dtype=np.float32)
            vec_cy = np.asarray(y0["dCy"],    dtype=np.float32)
            vec_g  = np.asarray(y0["dGamma"], dtype=np.float32)

            for g in range(256):
                # target이 NaN/inf면 해당 gray는 학습에서 제외
                if not (np.isfinite(vec_cx[g]) and np.isfinite(vec_cy[g]) and np.isfinite(vec_g[g])):
                    continue

                feat_row = self._build_features_for_gray_pred(X_dict=Xd, gray=g)
                X_rows.append(feat_row)
                y_dCx.append(float(vec_cx[g]))
                y_dCy.append(float(vec_cy[g]))
                y_dG.append(float(vec_g[g]))
                groups.append(pk)

        if not X_rows:
            logging.warning("[VACDataset] build_XYdataset_for_prediction_Y0: 유효한 샘플이 없습니다.")
            X_mat      = np.empty((0, 0), dtype=np.float32)
            y_cx_vec   = np.empty((0,), dtype=np.float32)
            y_cy_vec   = np.empty((0,), dtype=np.float32)
            y_g_vec    = np.empty((0,), dtype=np.float32)
            groups_arr = np.empty((0,), dtype=np.int64)
        else:
            X_mat      = np.vstack(X_rows).astype(np.float32)
            y_cx_vec   = np.asarray(y_dCx, dtype=np.float32)
            y_cy_vec   = np.asarray(y_dCy, dtype=np.float32)
            y_g_vec    = np.asarray(y_dG,  dtype=np.float32)
            groups_arr = np.asarray(groups, dtype=np.int64)

        Y0 = {
            "dCx":    y_cx_vec,
            "dCy":    y_cy_vec,
            "dGamma": y_g_vec,
        }
        return X_mat, Y0, groups_arr