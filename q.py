    # ================== Dataset 검증용 출력 ==================
    print("TEST")
    pd.set_option('display.max_columns', None)
    test_pk = [3008]
    dataset = VACDataset(test_pk, ref_pk=BYPASS_PK)

    X, y, groups = dataset.build_XY_dataset(
        target="y0",
        component="dGamma",
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W','R','G','B'),
    )

    print(f"\n[PK=3007 Y0 Dataset Preview]")
    print(f"X_mat shape: {X.shape}")   # (255, Dx)
    print(f"y_vec shape: {y.shape}")   # (255,)
    print("\n--- X_mat (first 3 rows) ---")
    print(pd.DataFrame(X[:3]))         # 앞부분 일부 확인
    print("\n--- y_vec (first 10 values) ---")
    print(y[:10])
    # ========================================================
