def train_Y0_models(dataset, save_dir, patterns=('W',), exclude_gray_for_cxcy=(0,5)):
    # 구버전 VACDataset은 component가 dGamma/dCx/dCy
    components = ["dGamma", "dCx", "dCy"]
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    for comp in components:
        print(f"\n=== Train Y0: {comp} ===")

        X_all, y_all, groups = dataset.build_XY_dataset(
            target="y0",
            component=comp,
            channels=channels,
            patterns=patterns,
        )

        # (선택) dCx/dCy는 회색 일부 제외를 dataset에서 이미 하고 있음
        artifacts = train_hybrid_regressor(X_all, y_all, groups, tag=f"Y0-{comp}")

        payload = {
            "target": {"type": "Y0-per-gray", "component": comp, "patterns": patterns},
            **artifacts,
            "feature_schema": {
                "desc": "ΔLUT(6ch) + meta + gray_norm + LUT_j",
                "channels": list(channels),
                "add_gray_norm": True,
                "add_LUT_j": True,
                "note": f"exclude_gray_for_cxcy={exclude_gray_for_cxcy} is applied inside VACDataset._build_XY0"
            }
        }
        path = os.path.join(save_dir, f"hybrid_{comp}_model.pkl")
        joblib.dump(payload, path, compress=("gzip", 3))
        print(f"📁 saved: {path}")
        
def train_Y1_model(dataset, save_dir, patterns=('W',)):
    print("\n=== Train Y1 ===")
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    X_all, y_all, groups = dataset.build_XY_dataset(
        target="y1",
        channels=channels,
        patterns=patterns,
    )

    # y_all에 NaN이 있을 수 있으니 방어 (권장)
    mask = np.isfinite(y_all)
    X_all = X_all[mask].astype(np.float32)
    y_all = y_all[mask].astype(np.float32)
    groups = groups[mask].astype(np.int64)

    artifacts = train_hybrid_regressor(X_all, y_all, groups, tag="Y1-slope")

    payload = {
        "target": {"type": "Y1-slope", "patterns": patterns},
        **artifacts,
        "feature_schema": {"desc": "segment-center gray ΔLUT(6ch)+meta+gray_norm+LUT_j"}
    }
    path = os.path.join(save_dir, "hybrid_Y1_slope_model.pkl")
    joblib.dump(payload, path, compress=("gzip", 3))
    print(f"📁 saved: {path}")
    
def train_Y2_model(dataset, save_dir):
    print("\n=== Train Y2 (delta_uv) ===")
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    X_all, y_all, groups = dataset.build_XY_dataset(
        target="y2",
        channels=channels,
    )

    mask = np.isfinite(y_all)
    X_all = X_all[mask].astype(np.float32)
    y_all = y_all[mask].astype(np.float32)
    groups = groups[mask].astype(np.int64)

    artifacts = train_hybrid_regressor(X_all, y_all, groups, tag="Y2-delta_uv")

    payload = {
        "target": {"type": "Y2-delta_uv"},
        **artifacts,
        "feature_schema": {"desc": "3 gray-triplets ΔLUT + meta + pattern one-hot (already inside dataset)"}
    }
    path = os.path.join(save_dir, "hybrid_Y2_delta_uv_model.pkl")
    joblib.dump(payload, path, compress=("gzip", 3))
    print(f"📁 saved: {path}")