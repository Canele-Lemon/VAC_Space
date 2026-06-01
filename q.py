# ================== Dataset 검증용 출력 ==================
print("\nTEST - Raw build_XY_dataset preview")
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

for target_pk in debug_target_pks:
    ref_pk = mapping.get_ref_pk(target_pk)

    print("\n" + "=" * 120)
    print(f"[DATASET PREVIEW] target_pk={target_pk}, auto ref_pk={ref_pk}")
    print("=" * 120)

    dataset_dbg = VACDataset(
        pk_list=[target_pk],
        set_mapping=mapping,
        drop_use_flag_N=False
    )

    if dataset_dbg.samples:
        print(
            f"[CHECK] sample pk={dataset_dbg.samples[0]['pk']}, "
            f"mapped ref_pk={dataset_dbg.samples[0]['ref_pk']}"
        )

    for comp in ("dGamma", "dCx", "dCy"):
        X, y, groups = dataset_dbg.build_XY_dataset(
            target="y0",
            component=comp,
            channels=channels,
            patterns=('W',),
        )

        print(f"\n[Y0-{comp}]")
        print(f"X_mat shape: {X.shape}")
        print(f"y_vec shape: {y.shape}")
        print(f"groups shape: {groups.shape}")
        print(f"unique groups: {np.unique(groups)}")

        print("\n--- X_mat first 5 rows ---")
        print(pd.DataFrame(X[:5]))

        print("\n--- y first 10 values ---")
        print(y[:10])

    X, y, groups = dataset_dbg.build_XY_dataset(
        target="y1",
        channels=channels,
        patterns=('W',),
    )

    print("\n[Y1-slope]")
    print(f"X_mat shape: {X.shape}")
    print(f"y_vec shape: {y.shape}")
    print(f"groups shape: {groups.shape}")
    print(f"unique groups: {np.unique(groups)}")
    print("\n--- X_mat first 5 rows ---")
    print(pd.DataFrame(X[:5]))
    print("\n--- y first 10 values ---")
    print(y[:10])

    X, y, groups = dataset_dbg.build_XY_dataset(
        target="y2",
        channels=channels,
    )

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