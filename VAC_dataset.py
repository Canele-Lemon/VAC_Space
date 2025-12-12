class VACDataset(Dataset):
    def __init__(self, pk_list, ref_pk=2582, drop_use_flag_N: bool = True):
        self.pk_list_all = list(pk_list)
        self.ref_pk = int(ref_pk)

        if drop_use_flag_N:
            self.pk_list = self._filter_by_use_flag(self.pk_list_all)
        else:
            self.pk_list = list(pk_list)

        if not self.pk_list:
            logging.warning("[VACDataset] 유효한 pk_list가 비어 있습니다.")

        self.samples = []
        self._collect()

    # -----------------------------
    # 공용 feature builder (meta 항상 포함)
    # -----------------------------
    def _build_features_for_gray(self, X_dict, gray: int, channels) -> np.ndarray:
        delta_lut = X_dict["lut_delta_raw"]  # dict: ch -> (256,)
        meta      = X_dict["meta"]
        j_map     = X_dict["mapping_j"]      # (256,)

        row = []
        # LUT delta
        for ch in channels:
            row.append(float(delta_lut[ch][gray]))

        # meta (always)
        row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # gray info
        row.append(gray / 255.0)
        row.append(float(int(j_map[gray])))

        return np.asarray(row, dtype=np.float32)

    def _build_pattern_onehot(self, pattern: str, pattern_order) -> np.ndarray:
        v = np.zeros(len(pattern_order), dtype=np.float32)
        if pattern in pattern_order:
            v[pattern_order.index(pattern)] = 1.0
        return v

    # -----------------------------
    # Y0 per-gray (dCx/dCy는 0~5 gray 제외)
    # -----------------------------
    def _build_XY0(
        self,
        component: str = "dGamma",
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W',),
        drop_gray_for_cxcy: int = 6,   # 0~5 제거하려면 6
    ):
        assert component in ("dGamma", "dCx", "dCy")

        X_rows, y_vals, groups = [], [], []
        gray_start = drop_gray_for_cxcy if component in ("dCx","dCy") else 0

        for s in self.samples:
            pk = s["pk"]
            Xd = s["X"]
            Yd = s["Y"]

            if "Y0" not in Yd:
                continue

            for p in patterns:
                if p not in Yd["Y0"]:
                    continue

                y_vec = Yd["Y0"][p][component]  # (256,)
                for g in range(gray_start, 256):
                    y_val = y_vec[g]
                    if not np.isfinite(y_val):
                        continue

                    feat = self._build_features_for_gray(Xd, g, channels)
                    # (선택) 패턴 one-hot을 Y0에도 붙이고 싶으면 아래 두 줄 사용
                    # pat_oh = self._build_pattern_onehot(p, pattern_order=list(patterns))
                    # feat = np.concatenate([feat, pat_oh], axis=0)

                    X_rows.append(feat)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

    # -----------------------------
    # Y1 per-segment (중앙 gray에서 X 뽑기)
    # -----------------------------
    def _build_XY1(
        self,
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W',),
        g_start=88,
        g_end=232,
        step=8,
    ):
        """
        - pk당 pattern당 18행
        - segment: 88-96, ..., 224-232
        - X: 중앙 gray = gs + step//2 (92,100,...) 의 ΔLUT + meta + gray_norm + LUT_j
        - y: 해당 segment slope (compute_Y1_struct에서 이미 abs 처리해둔 상태)
        """
        seg_starts = list(range(g_start, g_end, step))  # [88..224]
        mid_grays = [gs + (step // 2) for gs in seg_starts]  # [92,100,...,228] (18개)

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk = s["pk"]
            Xd = s["X"]
            Yd = s["Y"]

            if "Y1" not in Yd:
                continue

            for p in patterns:
                if p not in Yd["Y1"]:
                    continue
                slopes = np.asarray(Yd["Y1"][p], dtype=np.float32)  # (18,)

                for i, gmid in enumerate(mid_grays):
                    if i >= len(slopes):
                        break
                    y_val = slopes[i]
                    if not np.isfinite(y_val):
                        continue

                    feat = self._build_features_for_gray(Xd, int(gmid), channels)
                    # 필요하면 패턴 one-hot도 붙일 수 있음(기본은 W만 쓴다 가정)
                    X_rows.append(feat)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

    # -----------------------------
    # Y2 (B안): pk당 4행, pattern one-hot 포함
    # -----------------------------
    def _build_XY2(
        self,
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=("Darkskin", "Lightskin", "Asian", "Western"),
        gray_triplets=None,
        add_gray_loc_features: bool = True,  # 각 선택 gray의 (gray_norm, LUT_j)도 붙일지
    ):
        """
        B안:
        - 한 pk에서 4행 생성(4패턴)
        - 각 행 X는 '그 패턴에 해당하는 3개 gray'에서의 ΔLUT을 concat
        - y는 해당 패턴의 delta_uv (스칼라)
        - meta 항상 포함
        - pattern one-hot 항상 포함
        """

        if gray_triplets is None:
            gray_triplets = {
                "Darkskin":  (116, 80, 66),
                "Lightskin": (196, 150, 129),
                "Asian":     (196, 147, 118),
                "Western":   (183, 130, 93),
            }

        pattern_order = list(patterns)

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk = s["pk"]
            Xd = s["X"]
            Yd = s["Y"]

            if "Y2" not in Yd:
                continue

            for p in patterns:
                if p not in Yd["Y2"]:
                    continue

                y_val = float(Yd["Y2"][p])
                if not np.isfinite(y_val):
                    continue

                gs = gray_triplets.get(p, None)
                if gs is None:
                    continue

                # --- X 구성: (g1 LUT feats) + (g2 LUT feats) + (g3 LUT feats) + meta + pattern onehot ---
                feats = []

                for g in gs:
                    g = int(g)
                    # LUT delta
                    for ch in channels:
                        feats.append(float(Xd["lut_delta_raw"][ch][g]))

                    if add_gray_loc_features:
                        feats.append(g / 255.0)
                        feats.append(float(int(Xd["mapping_j"][g])))

                # meta 항상 포함
                meta = Xd["meta"]
                feats.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
                feats.append(float(meta["frame_rate"]))
                feats.append(float(meta["model_year"]))

                # pattern one-hot 항상 포함
                feats.extend(self._build_pattern_onehot(p, pattern_order).tolist())

                X_rows.append(np.asarray(feats, dtype=np.float32))
                y_vals.append(y_val)
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

    # -----------------------------
    # 통합 엔트리
    # -----------------------------
    def build_XY_dataset(
        self,
        target: str,
        component: str | None = None,
        channels=None,
        patterns=('W',),
    ):
        t = target.lower()

        if t == "y0":
            if component is None:
                raise ValueError("target='Y0'일 때 component('dGamma','dCx','dCy')가 필요합니다.")
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY0(component=component, channels=channels, patterns=patterns)

        if t == "y1":
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY1(channels=channels, patterns=patterns)

        if t == "y2":
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            # 여기서 patterns를 ('Darkskin','Lightskin','Asian','Western') 로 넘기면 pk당 4행
            return self._build_XY2(channels=channels, patterns=patterns)

        raise ValueError(f"Unknown target='{target}'. (지원: 'Y0','Y1','Y2')")