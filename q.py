if __name__ == "__main__":
    # 자코비안 학습 때 쓴 것과 동일하게
    pks = "2743-3002,!2743,!2744,!2984"
    from estimate_jacobian import parse_pks
    pk_list = parse_pks(pks)
    ref_pk = 2744

    jac_path = r"artifacts\jacobian_bundle_ref2744_lam0.001_dw900.0_gs30.0_2025....npy"

    debug_deltaG_sample_with_dataset(
        pk_list=pk_list,
        ref_pk=ref_pk,
        jacobian_npy_path=jac_path,
        target_gray=128,   # sanity test 하고 싶은 gray
        target_dG=50.0,    # ΔG_H = +50 근처
        tol_dG=2.0,        # ΔG_H 허용 오차
        tol_RB=5.0,        # R/B는 거의 안 건드린 샘플만 보도록
        max_show=3,
    )