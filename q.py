    def _build_XY0(
        self,
        component: str = "dGamma",
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W',),
        exclude_gray_for_cxcy=(0, 5),   # ✅ 추가: dCx/dCy일 때 제외할 gray 범위
    ):
        """
        Y0 예측용(X→ΔGamma/ΔCx/ΔCy) per-gray 데이터셋.
        - X: ΔLUT(지정 채널) + meta + gray_norm + LUT index
        - y: 선택한 component (ΔGamma/ΔCx/ΔCy), 지정된 패턴들(W/R/G/B)
        """
        assert component in ('dGamma', 'dCx', 'dCy')

        g_ex0, g_ex1 = exclude_gray_for_cxcy

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]
            Yd  = s["Y"]

            for p in patterns:
                if "Y0" not in Yd or p not in Yd["Y0"]:
                    continue

                y_vec = Yd["Y0"][p][component]  # (256,)
                for g in range(256):

                    # ✅ Cx/Cy는 0~5 gray 제외
                    if component in ("dCx", "dCy") and (g_ex0 <= g <= g_ex1):
                        continue

                    y_val = y_vec[g]
                    if not np.isfinite(y_val):
                        continue

                    feat_row = self._build_features_for_gray(
                        X_dict=Xd,
                        gray=g,
                        channels=channels,
                    )
                    X_rows.append(feat_row)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups
     
    def _build_XY1(
        self,
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W',),
        g_start=88, g_end=232, step=8,
    ):
        """
        Y1 예측용(X→slope segment) per-segment 데이터셋.
        - y: Y1 slope (segment별)
        - X: 해당 segment의 중앙 gray에서의 ΔRGB Low/High(지정 채널) + meta + gray_norm + LUT_j
        """
        seg_starts = list(range(g_start, g_end, step))  # [88,96,...,224] => 18개
        centers = [(gs + (gs + step)) // 2 for gs in seg_starts]  # [92,100,...,228]

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]
            Yd  = s["Y"]

            if "Y1" not in Yd:
                continue

            for p in patterns:
                if p not in Yd["Y1"]:
                    continue

                slopes = np.asarray(Yd["Y1"][p], dtype=np.float32)  # (18,)
                n_seg = min(len(slopes), len(centers))

                for i in range(n_seg):
                    y_val = slopes[i]
                    if not np.isfinite(y_val):
                        continue

                    cgray = int(centers[i])
                    if not (0 <= cgray <= 255):
                        continue

                    feat_row = self._build_features_for_gray(
                        X_dict=Xd,
                        gray=cgray,
                        channels=channels
                    )
                    X_rows.append(feat_row)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups
     
def _build_XY2(
    self,
    channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
    include_meta: bool = True,
    gray_map=None,
    y_patterns=('Darkskin','Lightskin','Asian','Western'),
):
    """
    Y2 예측용 (pk당 1행) 데이터셋.
    X: (지정 패턴별) (지정 gray 3개)에서의 ΔLUT[ch]를 펼친 벡터 + (선택)meta
    y: [Darkskin, Lightskin, Asian, Western] Δu'v' (Y2 값)

    Returns
    -------
    X_mat : (N, F)
    y_mat : (N, 4)
    groups: (N,)  # pk
    feature_names : list[str]  # 디버그/CSV용
    """
    if gray_map is None:
        gray_map = {
            "Darkskin":  [116, 80, 66],
            "Lightskin": [196, 150, 129],
            "Asian":     [196, 147, 118],
            "Western":   [183, 130, 93],
        }

    # feature name 만들기
    feature_names = []
    for ptn in y_patterns:
        gs = gray_map[ptn]
        for g in gs:
            for ch in channels:
                feature_names.append(f"{ptn}_g{g}_d{ch}")

    # meta feature name
    panel_dim = None
    if include_meta and self.samples:
        panel_dim = len(self.samples[0]["X"]["meta"]["panel_maker"])
        feature_names += [f"panel_{i}" for i in range(panel_dim)] + ["frame_rate", "model_year"]

    X_rows, y_rows, groups = [], [], []

    for s in self.samples:
        pk = s["pk"]
        Xd = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
        Yd = s["Y"]  # {"Y0":..., "Y1":..., "Y2":...}

        # ---- y 구성 (4개) ----
        if "Y2" not in Yd:
            continue
        y2 = Yd["Y2"]

        y_vec = []
        valid_y = True
        for ptn in y_patterns:
            v = y2.get(ptn, np.nan)
            if not np.isfinite(v):
                valid_y = False
                break
            y_vec.append(float(v))
        if not valid_y:
            continue

        # ---- X 구성 (72개 + meta) ----
        delta_lut = Xd["lut_delta_raw"]
        meta = Xd["meta"]

        row = []
        valid_x = True

        for ptn in y_patterns:
            for g in gray_map[ptn]:
                if not (0 <= g < 256):
                    valid_x = False
                    break
                for ch in channels:
                    v = delta_lut.get(ch, None)
                    if v is None:
                        valid_x = False
                        break
                    val = float(v[g])
                    if not np.isfinite(val):
                        valid_x = False
                        break
                    row.append(val)
            if not valid_x:
                break

        if not valid_x:
            continue

        if include_meta:
            row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
            row.append(float(meta["frame_rate"]))
            row.append(float(meta["model_year"]))

        X_rows.append(np.asarray(row, dtype=np.float32))
        y_rows.append(np.asarray(y_vec, dtype=np.float32))
        groups.append(pk)

    X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
    y_mat = np.vstack(y_rows).astype(np.float32) if y_rows else np.empty((0,0), np.float32)
    groups = np.asarray(groups, dtype=np.int64)

    return X_mat, y_mat, groups, feature_names