if __name__ == "__main__":
    import pandas as pd

    BYPASS_PK = 3007
    pk_list   = [3008]   # 우선 한 패널만 보는 게 보기 편함

    dataset = VACDataset(pk_list=pk_list, ref_pk=BYPASS_PK)

    # 내가 학습에 쓸 채널 정의
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    # Y0 - dGamma용 XY 생성
    X_dG, y_dG, grp_dG = dataset.build_XY_dataset(
        target="Y0",
        component="dGamma",
        channels=channels,
        patterns=('W',),   # 필요하면 ('W','R','G','B') 로도 가능
    )

    print("X_dG shape:", X_dG.shape)
    print("y_dG shape:", y_dG.shape)
    print("groups shape:", grp_dG.shape)

    # -------- 앞 몇 행만 'dataset 느낌'으로 보기 --------
    n_preview = min(30, X_dG.shape[0])  # 30행까지만

    # panel_maker one-hot 길이 가져오기 (첫 샘플 meta 이용)
    first_meta = dataset.samples[0]["X"]["meta"]
    panel_dim = len(first_meta["panel_maker"])

    feature_names = (
        [f"d{ch}" for ch in channels] +              # ΔLUT (Low/High 6채널)
        [f"panel_{i}" for i in range(panel_dim)] +   # panel maker one-hot
        ["frame_rate", "model_year", "gray_norm", "LUT_j"]
    )

    df = pd.DataFrame(X_dG[:n_preview, :], columns=feature_names)
    df["y"] = y_dG[:n_preview]
    df["pk_group"] = grp_dG[:n_preview]

    # 예쁘게 출력
    print("\n[PREVIEW] Y0-dGamma XY dataset (first rows)")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))