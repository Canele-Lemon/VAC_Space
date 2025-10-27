from scripts.VAC_dataset import VACDataset  # 경로에 맞춰 조정

def debug_train_flow_example():
    # 예시: 2444만으로 dGamma 학습셋 뽑기
    ds = VACDataset([2444])

    X, y, groups = ds.build_per_gray_y0_delta(
        component='dGamma',
        patterns=('W','R','G','B')
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("groups shape:", groups.shape)

    # 여기서 Ridge 학습
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge",  Ridge(alpha=1.0, random_state=42))
    ])
    model.fit(X, y)

    print("done. coef shape:", model.named_steps["ridge"].coef_.shape)