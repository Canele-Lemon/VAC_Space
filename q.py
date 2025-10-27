def debug_dump_single_pk(pk_debug=1411, knots_K=KNOTS, n_preview=5):
    # 1) knot 정의
    knots = make_knot_positions(K=knots_K)

    # 2) 해당 PK만으로 Dataset 생성
    ds = VACDataset([pk_debug])

    # 3) X, y 구성 (Gamma/Cx/Cy 중 하나씩 확인할 수도 있지만
    #    여기선 Gamma만 예시. 다른 것도 보면 comp 바꿔서 한 번 더 호출하면 돼요)
    X, y, feat_slices = build_dataset_Y0_abs(
        ds,
        knots,
        components=("Gamma",)  # ("Cx",) / ("Cy",) 로 바꿔서도 확인 가능
    )

    print(f"[DEBUG] pk={pk_debug}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("feature_slices:", feat_slices)

    # 4) 앞부분 몇 행만 preview
    for i in range(min(n_preview, len(y))):
        print(f"\n--- sample {i} ---")
        print(f"y[{i}] (target Gamma diff) = {y[i]}")
        print(f"X[{i}] (feature row) =\n{X[i]}")
        # 원하는 구간만 잘렸는지 보고 싶으면 예: high_R 구간만 따로 출력
        hr = X[i][feat_slices.high_R]
        hg = X[i][feat_slices.high_G]
        hb = X[i][feat_slices.high_B]
        meta_block = X[i][feat_slices.meta]
        gray_norm_val = X[i][feat_slices.gray]
        pattern_oh_block = X[i][feat_slices.pattern_oh]

        print(f"  high_R(phi-only)[len={len(hr)}]: {hr}")
        print(f"  high_G(phi-only)[len={len(hg)}]: {hg}")
        print(f"  high_B(phi-only)[len={len(hb)}]: {hb}")
        print(f"  meta                : {meta_block}")
        print(f"  gray_norm           : {gray_norm_val}")
        print(f"  pattern_onehot      : {pattern_oh_block}")

# -------------------------------------------------
# main() 대신 또는 main() 위에서 임시로 호출
# -------------------------------------------------
if __name__ == "__main__":
    # debug 전용 출력
    debug_dump_single_pk(pk_debug=1411, knots_K=KNOTS, n_preview=5)

    # 원래 학습 루틴을 돌릴 때는 아래를 다시 활성화하면 됩니다.
    # main()