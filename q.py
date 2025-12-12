def _build_XY0_for_jacobian_g(self, component='dGamma', exclude_gray_for_cxcy=(0, 5)):
    """
    자코비안 추정용 per-gray 데이터셋.
    - X: ΔLUT(High 3채널) + meta + gray_norm + LUT_j
    - y: dGamma / dCx / dCy (White 패턴, target - ref)

    규칙:
    - NaN인 y는 항상 제외
    - component가 dCx/dCy일 때만 gray 0~5 제외
    """
    assert component in ('dGamma', 'dCx', 'dCy')

    g_ex0, g_ex1 = exclude_gray_for_cxcy
    jac_channels = ('R_High', 'G_High', 'B_High')

    X_rows, y_vals, groups = [], [], []

    for s in self.samples:
        pk = s["pk"]
        Xd = s["X"]
        Yd = s["Y"]

        y_vec = Yd['Y0']['W'][component]  # (256,)

        for g in range(256):
            # dCx/dCy에서만 0~5 제외
            if component in ('dCx', 'dCy') and (g_ex0 <= g <= g_ex1):
                continue

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

    X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0, 0), np.float32)
    y_vec = np.asarray(y_vals, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    return X_mat, y_vec, groups
    
if __name__ == "__main__":
    pk_list = [3008]
    ref_pk = 3007
    ds = VACDataset(pk_list=pk_list, ref_pk=ref_pk)

    # 3개 컴포넌트 각각 뽑아서 row 수가 다르게 나오는지 확인
    for comp in ("dGamma", "dCx", "dCy"):
        X, y, grp = ds._build_XY0_for_jacobian_g(component=comp, exclude_gray_for_cxcy=(0, 5))

        print(f"\n[DEBUG] Jacobian raw XY - {comp}")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("groups shape:", grp.shape)

        if X.size == 0:
            print("(empty)")
            continue

        # feature name 구성
        meta = ds.samples[0]["X"]["meta"]
        K = len(meta["panel_maker"])
        feature_names = (
            ["dR_High", "dG_High", "dB_High"] +
            [f"panel_{i}" for i in range(K)] +
            ["frame_rate", "model_year", "gray_norm", "LUT_j"]
        )

        n = min(30, X.shape[0])
        df = pd.DataFrame(X[:n], columns=feature_names)
        df["gray_idx"] = np.clip(np.round(df["gray_norm"].to_numpy() * 255).astype(int), 0, 255)
        df["y"] = y[:n]
        df["pk_group"] = grp[:n]

        print(df.to_string(index=False, float_format=lambda v: f"{v: .6f}"))