def debug_dump_delta_training_rows():
    # 1) PK=2444만으로 dataset 구성
    ds = VACDataset(pk_list=[2444])

    # 2) ΔGamma 학습셋 생성
    X_mat, y_vec, groups = ds.build_per_gray_y0(component='dGamma', patterns=('W',))

    print("[DEBUG] dGamma dataset (ΔLUT -> ΔGamma)")
    print("X_mat shape:", X_mat.shape)   # 예상: (유효 gray 수, feature_dim)
    print("y_vec shape:", y_vec.shape)   # 예상: (유효 gray 수,)
    print("groups shape:", groups.shape) # 모든 값이 2444일 것

    if X_mat.shape[0] == 0:
        print("No valid samples (all NaN?). Check measurement data or gamma calc.")
        return

    # panel one-hot 길이 파악
    panel_len = len(ds.samples[0]["X"]["meta"]["panel_maker"])

    # 3) 앞에서 몇 개만 출력
    for i in range(min(5, X_mat.shape[0])):
        print(f"\n--- sample {i} ---")
        print("pk:", groups[i])
        print("y (ΔGamma vs ref):", y_vec[i])

        feat = X_mat[i]

        # feat layout:
        # [ΔR_Low, ΔR_High, ΔG_Low, ΔG_High, ΔB_Low, ΔB_High,
        #  panel_onehot..., frame_rate, model_year,
        #  gray_norm,
        #  pattern_onehot(4)]

        delta_lut_part = feat[:6]
        panel_oh       = feat[6 : 6+panel_len]
        frame_rate     = feat[6+panel_len]
        model_year     = feat[6+panel_len+1]
        gray_norm      = feat[6+panel_len+2]
        pattern_onehot = feat[6+panel_len+3 : 6+panel_len+7]

        print("ΔLUT[0:6]             :", delta_lut_part)
        print("panel_onehot          :", panel_oh)
        print("frame_rate            :", frame_rate)
        print("model_year            :", model_year)
        print("gray_norm             :", gray_norm)
        print("pattern_onehot(WRGB)  :", pattern_onehot)

    print("\n[CHECK]")
    print("- ΔLUT[0:6]는 prepare_X_delta()에서 본 delta lut 값과 동일해야 합니다 (같은 gray 인덱스).")
    print("- y는 ΔGamma = target_gamma - ref_gamma 이므로 0에 가까우면 레퍼런스와 유사.")
    print("- gray_norm가 0.5 근처면 gray≈128 정도 샘플일 거고, pattern_onehot이 [1,0,0,0]이면 'W' 패턴입니다.")


if __name__ == "__main__":
    debug_dump_delta_training_rows()