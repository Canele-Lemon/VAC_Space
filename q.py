def main():
    # ========================================================================================= #
    #                                         변수 지정                                        
    # ========================================================================================= #
    # 1. 학습에 사용할 PK(s)
    pks = "2456-2677,!2456"
    pk_list = parse_pks(pks)
    
    # 2. REF LUT PK
    ref_pk = 2582
    
    # 3. lam
    lam = 1e-3
    
    # 4. delta-window
    delta_window = 50
    
    # 5. gauss-sigma
    gauss_sigma = 30
    
    # 6. min_samples
    min_samples = 3
    # ========================================================================================= #

    jac, df = estimate_jacobians_per_gray(
        pk_list=pk_list, 
        ref_pk=ref_pk, 
        lam=lam, 
        delta_window=delta_window, 
        gauss_sigma=gauss_sigma,
        min_samples=min_samples
        )
    out_csv, out_npy = make_default_paths(ref_pk=ref_pk, lam=lam, delta_window=delta_window, gauss_sigma=gauss_sigma)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    
    # NPY 저장 (J 번들)
    J_dense = np.full((256, 3, 3), np.nan, dtype=np.float32)
    n_arr   = np.zeros(256, dtype=np.int32)
    condArr = np.full(256, np.nan, dtype=np.float32)
    for g, payload in jac.items():
        J_dense[g, :, :] = payload["J"]
        n_arr[g] = int(payload["n"])
        condArr[g] = float(payload["cond"])

    bundle = {
        "J": J_dense,
        "n": n_arr,
        "cond": condArr,
        "ref_pk": ref_pk,
        "lam": 1e-3,
        "delta_window": 50,
        "gauss_sigma": 30,
        "pk_list": pk_list,
        "gray_used": [2, 253],
        "schema": "J[gray, out(Cx,Cy,Gamma), in(R_High,G_High,B_High)]",
    }
    np.save(out_npy, bundle, allow_pickle=True)

    print(f"[OK] CSV saved -> {out_csv}")
    print(f"[OK] NPY saved -> {out_npy}")

    # 미리보기
    for g in (0, 32, 128, 255):
        if 0 <= g < 256 and np.isfinite(J_dense[g]).any():
            print(f"\n[g={g}] n={n_arr[g]}, cond={condArr[g]:.2e}")
            print(J_dense[g])
        else:
            print(f"\n[g={g}] no estimate (NaN or insufficient samples)")
