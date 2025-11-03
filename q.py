if __name__ == "__main__":
    ds = VACDataset(pk_list=[2635], ref_vac_info_pk=2582)

    X_mat, y_vec, groups = ds.build_white_y0_delta(component='dCx',
                                                   feature_channels=('R_High','G_High','B_High'))

    print("X_mat shape:", X_mat.shape)
    print("y_vec shape:", y_vec.shape)

    # 앞 몇 행만 출력
    for i in range(5):
        print(f"\n--- row {i} ---")
        print("X:", X_mat[i])
        print("y:", y_vec[i])