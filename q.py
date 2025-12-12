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