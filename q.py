"feature_schema": {
    "desc": "segment-center gray ΔLUT(6ch) + panel_maker_onehot + frame_rate + gray_norm + LUT_j",
    "channels": list(channels),
    "meta": ["panel_maker_onehot", "frame_rate"],
    "model_year_used": False
}

"feature_schema": {
    "desc": "3 gray-triplets ΔLUT + panel_maker_onehot + frame_rate + pattern one-hot",
    "channels": list(channels),
    "meta": ["panel_maker_onehot", "frame_rate"],
    "model_year_used": False
}

if X_all.size == 0 or y_all.size == 0:
    raise ValueError(f"[{tag}] Empty dataset. X={X_all.shape}, y={y_all.shape}")

if len(np.unique(groups)) < 4:
    raise ValueError(f"[{tag}] Too few groups for GroupKFold/holdout. unique groups={len(np.unique(groups))}")
    
    .

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

    # -------------------------------------------
    # 3) 저장 경로
    # -------------------------------------------
    save_dir = os.path.dirname(__file__)

    # -------------------------------------------
    # 4) 학습 실행
    # -------------------------------------------
    train_Y0_models(dataset, save_dir, patterns=('W',))
    train_Y1_model(dataset, save_dir, patterns=('W',))
    train_Y2_model(dataset, save_dir)

    print("\n✅ ALL DONE.")