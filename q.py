def debug_dataset_summary(pk_list, ref_pk):
    X, Y0, groups, idx_gray, ds = build_white_X_Y0(pk_list, ref_pk)

    dRGB = X[:, :3]
    gray_norm = X[:, idx_gray]
    gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    print("\n================ DATASET SUMMARY ================")
    print(f"ref_pk = {ref_pk}")
    print(f"pk_list size = {len(pk_list)} (unique in rows={len(set(groups.tolist()))})")
    print(f"X shape  = {X.shape}   (expected: N x >=3)")
    print(f"Y0 shape = {Y0.shape}  (expected: N x 3)")
    print(f"idx_gray = {idx_gray}")

    # NaN 체크
    nanX = np.isnan(X).any(axis=1).sum()
    nanY = np.isnan(Y0).any(axis=1).sum()
    print(f"rows with any NaN: X={nanX}, Y={nanY}")

    # gray 분포
    cnt = np.bincount(gray_idx, minlength=256)
    print(f"gray count: min={cnt.min()}, max={cnt.max()}, nonzero_grays={(cnt>0).sum()}/256")
    print("sample counts at some grays:", {g:int(cnt[g]) for g in [0,1,2,6,32,128,247,248,253,254,255]})

    # ΔRGB magnitude 분포
    mag = np.linalg.norm(dRGB, axis=1)
    print(f"|dRGB|: min={np.nanmin(mag):.3f}, mean={np.nanmean(mag):.3f}, max={np.nanmax(mag):.3f}")
    for p in [50, 90, 95, 99]:
        print(f"  percentile {p}% = {np.nanpercentile(mag, p):.3f}")

    # target(Y) 분포
    absY = np.abs(Y0)
    print(f"|dY| mean: dCx={np.nanmean(absY[:,0]):.6f}, dCy={np.nanmean(absY[:,1]):.6f}, dG={np.nanmean(absY[:,2]):.6f}")

    # pk별로 gray당 샘플 수가 1인지 확인
    # (대부분 (pk,gray)=1개면 jacobian per-gray 학습이 어려움)
    from collections import Counter
    key_counts = Counter(zip(groups.tolist(), gray_idx.tolist()))
    cvals = np.array(list(key_counts.values()), dtype=int)
    print(f"(pk,gray) multiplicity: min={cvals.min()}, median={np.median(cvals)}, max={cvals.max()}")
    print("=================================================\n")