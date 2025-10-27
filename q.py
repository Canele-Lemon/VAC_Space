def build_A_for_component(artifacts: dict, comp: str, L=256) -> np.ndarray:
    """
    ΔY ≈ A Δh
    여기서 Δh는 [R_high_knots(K), G_high_knots(K), B_high_knots(K),
                 R_low_knots(K),  G_low_knots(K),  B_low_knots(K)]
    A shape = (L, 6K)
    """
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    comp_obj = artifacts["components"][comp]
    coef = np.asarray(comp_obj["coef"], dtype=np.float32)
    scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)

    s = comp_obj["feature_slices"]
    # slice 정보 다 있음
    beta_high_R = coef[s["high_R"][0]:s["high_R"][1]] / np.maximum(scale[s["high_R"][0]:s["high_R"][1]], 1e-12)
    beta_high_G = coef[s["high_G"][0]:s["high_G"][1]] / np.maximum(scale[s["high_G"][0]:s["high_G"][1]], 1e-12)
    beta_high_B = coef[s["high_B"][0]:s["high_B"][1]] / np.maximum(scale[s["high_B"][0]:s["high_B"][1]], 1e-12)

    beta_low_R  = coef[s["low_R"][0]:s["low_R"][1]]   / np.maximum(scale[s["low_R"][0]:s["low_R"][1]],   1e-12)
    beta_low_G  = coef[s["low_G"][0]:s["low_G"][1]]   / np.maximum(scale[s["low_G"][0]:s["low_G"][1]],   1e-12)
    beta_low_B  = coef[s["low_B"][0]:s["low_B"][1]]   / np.maximum(scale[s["low_B"][0]:s["low_B"][1]],   1e-12)

    Phi = stack_basis_all_grays(knots, L=L)  # (L,K), phi(g) basis over gray

    A_high_R = Phi * beta_high_R.reshape(1, -1)
    A_high_G = Phi * beta_high_G.reshape(1, -1)
    A_high_B = Phi * beta_high_B.reshape(1, -1)

    A_low_R  = Phi * beta_low_R.reshape(1, -1)
    A_low_G  = Phi * beta_low_G.reshape(1, -1)
    A_low_B  = Phi * beta_low_B.reshape(1, -1)

    A = np.hstack([A_high_R, A_high_G, A_high_B,
                   A_low_R,  A_low_G,  A_low_B]).astype(np.float32)
    return A