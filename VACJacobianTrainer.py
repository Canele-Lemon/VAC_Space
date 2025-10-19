def build_A_for_component(artifacts: dict, comp: str, L=256) -> np.ndarray:
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    comp_obj = artifacts["components"][comp]
    coef  = np.asarray(comp_obj["coef"], dtype=np.float32)
    scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)  # ★

    s = comp_obj["feature_slices"]
    s_high_R = slice(s["high_R"][0], s["high_R"][1])
    s_high_G = slice(s["high_G"][0], s["high_G"][1])
    s_high_B = slice(s["high_B"][0], s["high_B"][1])

    # ★ 표준화 역적용: beta_eff = coef / scale
    beta_R = coef[s_high_R] / np.maximum(scale[s_high_R], 1e-12)
    beta_G = coef[s_high_G] / np.maximum(scale[s_high_G], 1e-12)
    beta_B = coef[s_high_B] / np.maximum(scale[s_high_B], 1e-12)

    Phi = stack_basis_all_grays(knots, L=L)  # (L,K)

    A_R = Phi * beta_R.reshape(1, -1)
    A_G = Phi * beta_G.reshape(1, -1)
    A_B = Phi * beta_B.reshape(1, -1)
    return np.hstack([A_R, A_G, A_B]).astype(np.float32)
    
    for g in range(256):
    val = arr[g]
    if comp == 'Gamma' and (g < 1 or g > 254):  # ★ 경계 제거
        continue
    if not np.isfinite(val):
        continue
    ...
    def _build_A_from_artifacts(self, artifacts, comp: str):
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    comp_obj = artifacts["components"][comp]
    coef  = np.asarray(comp_obj["coef"], dtype=np.float32)
    scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)  # ★

    s = comp_obj["feature_slices"]
    s_high_R = slice(s["high_R"][0], s["high_R"][1])
    s_high_G = slice(s["high_G"][0], s["high_G"][1])
    s_high_B = slice(s["high_B"][0], s["high_B"][1])

    beta_R = coef[s_high_R] / np.maximum(scale[s_high_R], 1e-12)  # ★
    beta_G = coef[s_high_G] / np.maximum(scale[s_high_G], 1e-12)  # ★
    beta_B = coef[s_high_B] / np.maximum(scale[s_high_B], 1e-12)  # ★

    Phi = self._stack_basis(knots, L=256)  # (256,K)

    A_R = Phi * beta_R.reshape(1, -1)
    A_G = Phi * beta_G.reshape(1, -1)
    A_B = Phi * beta_B.reshape(1, -1)

    A = np.hstack([A_R, A_G, A_B]).astype(np.float32)  # (256, 3K)
    logging.info(f"[Jacobian] {comp} A 행렬 shape: {A.shape}")
    return A
    
    
    