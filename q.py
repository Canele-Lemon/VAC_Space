# ================== Dataset 검증용 출력 ==================    
print("\nTEST - XY dataset preview from actual training dataset")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

debug_target_pks = [
    4300,  # 50QNED85 INX
    4000,  # 50QNED85 HKC
    3700,  # 43UT80 CSOT
    3400,  # 43NANO80 HKC
    3100,  # 50UB85 INX
]

channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

sample_ref_map = {
    int(s["pk"]): int(s["ref_pk"])
    for s in dataset.samples
}

for target_pk in debug_target_pks:
    print("\n" + "=" * 120)
    print(f"[DATASET PREVIEW] target_pk={target_pk}, ref_pk={sample_ref_map.get(target_pk)}")
    print("=" * 120)

    if target_pk not in sample_ref_map:
        print(f"[SKIP] target_pk={target_pk} is not included in actual training dataset.")
        continue

    for comp in ("dGamma", "dCx", "dCy"):
        X_all, y_all, groups_all = dataset.build_XY_dataset(
            target="y0",
            component=comp,
            channels=channels,
            patterns=('W',),
        )

        mask = groups_all == target_pk
        X = X_all[mask]
        y = y_all[mask]
        groups = groups_all[mask]

        print(f"\n[Y0-{comp}]")
        print(f"X_mat shape: {X.shape}")
        print(f"y_vec shape: {y.shape}")
        print(f"groups shape: {groups.shape}")
        print(f"unique groups: {np.unique(groups)}")

        print("\n--- X_mat first 5 rows ---")
        print(pd.DataFrame(X[:5]))

        print("\n--- y first 10 values ---")
        print(y[:10])

    X_all, y_all, groups_all = dataset.build_XY_dataset(
        target="y1",
        channels=channels,
        patterns=('W',),
    )

    mask = groups_all == target_pk
    X = X_all[mask]
    y = y_all[mask]
    groups = groups_all[mask]

    print("\n[Y1-slope]")
    print(f"X_mat shape: {X.shape}")
    print(f"y_vec shape: {y.shape}")
    print(f"groups shape: {groups.shape}")
    print(f"unique groups: {np.unique(groups)}")
    print("\n--- X_mat first 5 rows ---")
    print(pd.DataFrame(X[:5]))
    print("\n--- y first 10 values ---")
    print(y[:10])

    X_all, y_all, groups_all = dataset.build_XY_dataset(
        target="y2",
        channels=channels,
    )

    mask = groups_all == target_pk
    X = X_all[mask]
    y = y_all[mask]
    groups = groups_all[mask]

    print("\n[Y2-delta_uv]")
    print(f"X_mat shape: {X.shape}")
    print(f"y_vec shape: {y.shape}")
    print(f"groups shape: {groups.shape}")
    print(f"unique groups: {np.unique(groups)}")
    print("\n--- X_mat first 5 rows ---")
    print(pd.DataFrame(X[:5]))
    print("\n--- y first 10 values ---")
    print(y[:10])
# ========================================================