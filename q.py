def main():
    # -------------------------------------------
    # 1) vac_set_mapping.csv 기준 학습 PK 구성
    # -------------------------------------------
    mapping = VACSetMapping()
    TARGET_PK_LIST = mapping.build_target_pk_list()

    print(f"▶ Train with {len(TARGET_PK_LIST)} PKs")
    print(f"▶ Mapping file: {mapping.csv_path}")
    print(mapping.df.to_string(index=False))

    # -------------------------------------------
    # 2) 데이터셋 생성
    #    각 PK별 ref_pk는 VACDataset 내부에서 mapping 기준으로 선택
    # -------------------------------------------
    dataset = VACDataset(
        pk_list=TARGET_PK_LIST,
        set_mapping=mapping,
        drop_use_flag_N=True
    )

    print(f"▶ Valid PKs after Use_Flag filtering: {len(dataset.pk_list)}")
    print(f"▶ Collected samples: {len(dataset.samples)}")


    

    # ================== 학습 실행 ================== 
    # save_dir = os.path.dirname(__file__)
    
    # train_Y0_models(dataset, save_dir, patterns=('W',))
    # train_Y1_model(dataset, save_dir, patterns=('W',))
    # train_Y2_model(dataset, save_dir)

    # print("\n✅ ALL DONE.")
    # ========================================================
    
    
    
    # ================== Dataset 검증용 출력 ==================    
    print("\nTEST - XY dataset preview")
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

이렇게 수정하고 실행했는데

PS D:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project> & C:/python310/python.exe "d:/00 업무/00 가상화기술/25Y/00 색시야각 보상 최적화/VAC algorithm/VAC_Optimization_Project/src/modeling/train_model.py"
▶ Train with 1560 PKs
▶ Mapping file: d:\00 업무\00 가상화기술\25Y\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\data\vac_set_mapping.csv
 pk_start  pk_end  ref_pk  base_pk model_name panel_maker  frame_rate model_year                     memo
     4254    4566    4254     4255   50QNED85         INX         120        Y25    ref_pk=4565,4348,4336
     3943    4253    3943     3944   50QNED85     HKC(H2)         120        Y25              ref_pk=4163
     3631    3942    3631     3632     43UT80  CSOT(CSPI)          60        Y24         ref_pk=3931,3940
     3320    3630    3320     3321   43NANO80     HKC(H2)          60        Y25              ref_pk=3553
     3007    3319    3007     3008     50UB85         INX          60        Y26 ref_pk=3317/base_pk=3318
INFO:root:[VACDataset] Use_Flag='N' 이라 제외된 PK: [3317, 3318, 3553, 3931, 3940, 4163, 4336, 4348, 4565]

여기서 멈춘 후 엄청 오래 걸리네요... target_pk만 디버깅하는건데 왜이렇게 오래 걸리는거죠
