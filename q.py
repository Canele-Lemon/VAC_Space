    def _build_features_for_gray(self, X_dict, gray: int, channels) -> np.ndarray:
        """
        한 행(feature) 구성:
        [ΔR_High, ΔG_High, ΔB_High, panel_maker(one-hot), frame_rate, model_year, gray_norm(=g/255), LUT_index_j(g)]

        - ΔR_High, ΔG_High, ΔB_High: LUT index 매핑 포인트 기준 (LUT 값)-(ref LUT 값). normalize 안함!
        - panel_maker(one-hot), frame_rate, model_year: meta에서 그대로 가져옵니다.
        - LUT_index_j: mapping_j[gray] (0..4095), raw 그대로
        """
        # 1) 소스 참조
        delta_lut = X_dict["lut_delta_raw"]   # dict: ch -> (256,) float32 (raw delta at mapped indices)
        meta      = X_dict["meta"]            # dict: panel_maker(one-hot), frame_rate, model_year
        j_map     = X_dict["mapping_j"]       # (256,) int32, gray -> LUT index(0..4095)

        # 2) 채널 부분: 지정된 feature_channels만 사용 (보통 High 3채널)
        row = []
        for ch in channels:
            row.append(float(delta_lut[ch][gray]))   # raw delta (정규화 없음)

        # 3) 메타 부착
        row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # 4) gray 위치 정보
        row.append(gray / 255.0)                    # gray_norm

        # 5) LUT 물리 인덱스(매핑) 정보
        j_idx = int(j_map[gray])                    # 0..4095, raw
        row.append(float(j_idx))

        return np.asarray(row, dtype=np.float32)

    def _build_XY0_for_jacobian_g(self, component='dGamma'):
        """
        - X: raw ΔLUT(High 3채널) + meta + gray_norm + LUT index
        - y: dGamma / dCx / dCy (White 패턴, target - ref)
        """
        assert component in ('dGamma','dCx','dCy')

        jac_channels = ('R_High', 'G_High', 'B_High')
        
        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
            Yd  = s["Y"]  # {"Y0": {...}, ...}

            p = 'W'
            y_vec = Yd['Y0'][p][component]  # (256,)
            for g in range(256):
                y_val = y_vec[g]
                if not np.isfinite(y_val):
                    continue
                feat_row = self._build_features_for_gray(
                    X_dict=Xd, 
                    gray=g,
                    channels=jac_channels
                )
                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

여기서 데이터셋을 준비할 때 Cx, Cy는 0~5gray 까지 불안정하기 때문에 Gamma 에 대해서만 만들거에요 (NaN제외)
수정 후 우선 VACDataset에서 데이터확인용 출력해볼게요.
