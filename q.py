def _build_XY2(
    self,
    channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
    patterns=("Darkskin", "Lightskin", "Asian", "Western"),
    gray_triplets=None,
):
    """
    B안:
    - pk당 4행 (Darkskin / Lightskin / Asian / Western)
    - X:
        [g1의 ΔLUT + g1 gray_norm + g1 LUT_j,
         g2의 ΔLUT + g2 gray_norm + g2 LUT_j,
         g3의 ΔLUT + g3 gray_norm + g3 LUT_j,
         meta,
         pattern one-hot]
    - y: 해당 패턴의 delta_uv
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

            gs = gray_triplets.get(p)
            if gs is None:
                continue

            feats = []

            for g in gs:
                g = int(g)

                # ΔLUT
                for ch in channels:
                    feats.append(float(Xd["lut_delta_raw"][ch][g]))

                # 항상 gray 위치 정보 포함
                feats.append(g / 255.0)                   # gray_norm
                feats.append(float(Xd["mapping_j"][g]))  # LUT_j

            # meta (항상 포함)
            meta = Xd["meta"]
            feats.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
            feats.append(float(meta["frame_rate"]))
            feats.append(float(meta["model_year"]))

            # pattern one-hot (항상 포함)
            feats.extend(self._build_pattern_onehot(p, pattern_order).tolist())

            X_rows.append(np.asarray(feats, dtype=np.float32))
            y_vals.append(y_val)
            groups.append(pk)

    X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
    y_vec = np.asarray(y_vals, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)

    return X_mat, y_vec, groups
    
def preview_xy_dataset(X, y, groups, feature_names, title="", n=30):
    import numpy as np
    import pandas as pd

    print(f"\n[PREVIEW] {title}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("groups shape:", groups.shape)

    if X.size == 0:
        print("(empty)")
        return

    n = min(n, X.shape[0])
    df = pd.DataFrame(X[:n], columns=feature_names)
    df["y"] = y[:n]
    df["pk_group"] = groups[:n]
    print(df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))


if __name__ == "__main__":
    pk_list = [3008]
    BYPASS_PK = 3007

    dataset = VACDataset(pk_list=pk_list, ref_pk=BYPASS_PK)

    # -------- Y1 dataset 생성 --------
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    X_y1, y_y1, grp_y1 = dataset.build_XY_dataset(
        target="Y1",
        channels=channels,
        patterns=('W',),
    )

    # -------- feature name 생성 (Y1: 중앙 gray 1개 사용 가정) --------
    first_meta = dataset.samples[0]["X"]["meta"]
    panel_dim = len(first_meta["panel_maker"])

    # (중앙 gray 1개에 대해) [dLUT 6] + [gray_norm, LUT_j]
    feat_names_y1 = []
    feat_names_y1 += [f"d{ch}" for ch in channels]
    feat_names_y1 += ["gray_norm", "LUT_j"]

    # meta
    feat_names_y1 += [f"panel_{i}" for i in range(panel_dim)]
    feat_names_y1 += ["frame_rate", "model_year"]

    # (선택) 만약 _build_XY1에서 seg_idx 같은 걸 붙였다면 여기에 이름 추가
    # feat_names_y1 += ["seg_idx"]  # <- 실제로 X에 포함시켰을 때만!

    preview_xy_dataset(
        X_y1, y_y1, grp_y1,
        feature_names=feat_names_y1,
        title="Y1 slope XY dataset",
        n=30
    )

if __name__ == "__main__":
    pk_list = [3008]
    BYPASS_PK = 3007

    dataset = VACDataset(pk_list=pk_list, ref_pk=BYPASS_PK)

    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
    patterns = ("Darkskin", "Lightskin", "Asian", "Western")

    X_y2, y_y2, grp_y2 = dataset.build_XY_dataset(
        target="Y2",
        channels=channels,
    )

    # ---- feature names 만들기 ----
    first_meta = dataset.samples[0]["X"]["meta"]
    panel_dim = len(first_meta["panel_maker"])

    # Y2는 gray 3개(triplet) → 각 gray마다 (dLUT 6 + gray_norm + LUT_j)
    feat_names_y2 = []
    for k in range(1, 4):  # g1,g2,g3
        feat_names_y2 += [f"g{k}_d{ch}" for ch in channels]
        feat_names_y2 += [f"g{k}_gray_norm", f"g{k}_LUT_j"]

    # meta
    feat_names_y2 += [f"panel_{i}" for i in range(panel_dim)]
    feat_names_y2 += ["frame_rate", "model_year"]

    # pattern one-hot(4)
    feat_names_y2 += [f"pat_{p}" for p in patterns]

    preview_xy_dataset(
        X_y2, y_y2, grp_y2,
        feature_names=feat_names_y2,
        title="Y2 delta_uv XY dataset (4 patches)",
        n=20
    )