def build_white_X_Y3(pk_list, ref_pk):
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_pk)

    X_cx, y_cx, g_cx = ds.build_white_y0_delta(component='dCx')
    X_cy, y_cy, g_cy = ds.build_white_y0_delta(component='dCy')
    X_ga, y_ga, g_ga = ds.build_white_y0_delta(component='dGamma')

    # 1) 기본 형태/그룹 정합 체크
    if not (X_cx.shape == X_cy.shape == X_ga.shape):
        raise RuntimeError("X 행렬 형태가 일치하지 않습니다 (dCx/dCy/dGamma).")
    if not (np.all(g_cx == g_cy) and np.all(g_cx == g_ga)):
        raise RuntimeError("groups 순서가 일치하지 않습니다.")

    # 2) gray index 복원
    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])  # panel one-hot 길이
    idx_gray = 3 + K + 2   # [ΔR,ΔG,ΔB]=3 + panel(K) + frame + year → gray_norm
    X = X_cx.astype(np.float32)
    gray_norm = X[:, idx_gray]
    gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    # 3) 유효 마스크: gray∈{1..254} & 3타깃 모두 finite
    Y3 = np.stack([y_cx, y_cy, y_ga], axis=1).astype(np.float32)
    finite_mask = np.isfinite(Y3).all(axis=1)
    core_mask = (gray_idx >= 1) & (gray_idx <= 254) & finite_mask

    # 4) 필터 적용
    X = X[core_mask]
    Y3 = Y3[core_mask]
    groups = g_cx[core_mask]

    return X, Y3, groups, idx_gray, ds