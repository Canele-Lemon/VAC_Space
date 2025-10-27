def _build_A_from_artifacts(self, artifacts, comp: str):
    """
    저장된 자코비안 pkl로부터 A 행렬 (ΔY ≈ A·Δh) 복원
    이제 Δh = [ΔR_Low_knots, ΔG_Low_knots, ΔB_Low_knots,
               ΔR_High_knots,ΔG_High_knots,ΔB_High_knots] (총 6*K)
    반환 A shape: (256, 6*K)
    """
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    comp_obj = artifacts["components"][comp]

    coef  = np.asarray(comp_obj["coef"], dtype=np.float32)
    scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)

    s = comp_obj["feature_slices"]
    # 6채널 모두
    slices = [
        ("low_R",  "R_Low"),
        ("low_G",  "G_Low"),
        ("low_B",  "B_Low"),
        ("high_R", "R_High"),
        ("high_G", "G_High"),
        ("high_B", "B_High"),
    ]

    Phi = self._stack_basis(knots, L=256)    # (256,K)

    A_blocks = []
    for key_slice, _pretty_name in slices:
        sl = slice(s[key_slice][0], s[key_slice][1])   # e.g. (0,33), (33,66), ...
        beta = coef[sl] / np.maximum(scale[sl], 1e-12)  # (K,)
        A_ch = Phi * beta.reshape(1, -1)                # (256,K)
        A_blocks.append(A_ch)

    A = np.hstack(A_blocks).astype(np.float32)          # (256, 6K)
    logging.info(f"[Jacobian] {comp} A 행렬 shape: {A.shape}") 
    return A