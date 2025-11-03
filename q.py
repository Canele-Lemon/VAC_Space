def build_white_X_Y3(pk_list, ref_pk):
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_pk)

    X_cx, y_cx, g_cx = ds.build_white_y0_delta(component='dCx')
    X_cy, y_cy, g_cy = ds.build_white_y0_delta(component='dCy')
    X_ga, y_ga, g_ga = ds.build_white_y0_delta(component='dGamma')

    # gray_norm 컬럼 인덱스
    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])
    idx_gray_cx = 3 + K + 2
    idx_gray_cy = 3 + K + 2
    idx_gray_ga = 3 + K + 2

    def keys_from(X, groups, idx_gray):
        gray_norm = X[:, idx_gray]
        gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)
        # 키 = (pk, gray) 튜플 리스트
        return [(int(pk), int(g)) for pk, g in zip(groups.astype(np.int64), gray_idx)]

    keys_cx = keys_from(X_cx, g_cx, idx_gray_cx)
    keys_cy = keys_from(X_cy, g_cy, idx_gray_cy)
    keys_ga = keys_from(X_ga, g_ga, idx_gray_ga)

    # 세 집합의 교집합 (튜플 기반이므로 안전)
    common = sorted(set(keys_cx) & set(keys_cy) & set(keys_ga))
    if not common:
        raise RuntimeError("세 컴포넌트의 (pk, gray) 교집합이 비어 있습니다.")

    # 공통 키 → 각 배열의 인덱스 매핑
    idx_map_cx = {k:i for i,k in enumerate(keys_cx)}
    idx_map_cy = {k:i for i,k in enumerate(keys_cy)}
    idx_map_ga = {k:i for i,k in enumerate(keys_ga)}

    idx_cx = np.array([idx_map_cx[k] for k in common], dtype=np.int64)
    idx_cy = np.array([idx_map_cy[k] for k in common], dtype=np.int64)
    idx_ga = np.array([idx_map_ga[k] for k in common], dtype=np.int64)

    # 동일 순서로 정렬
    X = X_cx[idx_cx].astype(np.float32)  # X는 동일 형태라 CX 기준으로 택일
    y_cx_sel = y_cx[idx_cx].astype(np.float32)
    y_cy_sel = y_cy[idx_cy].astype(np.float32)
    y_ga_sel = y_ga[idx_ga].astype(np.float32)

    # g=2..253 & 세 타깃 finite
    gray_norm = X[:, 3 + K + 2]
    gray_idx  = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)
    Y3 = np.stack([y_cx_sel, y_cy_sel, y_ga_sel], axis=1)

    core_mask = (gray_idx >= 2) & (gray_idx <= 253) & np.isfinite(Y3).all(axis=1)
    X  = X[core_mask]
    Y3 = Y3[core_mask]

    # groups는 공통 키의 pk를 사용, core_mask로 다시 필터
    groups_all = np.array([pk for pk, _g in common], dtype=np.int64)
    groups = groups_all[core_mask]

    idx_gray = 3 + K + 2
    return X, Y3, groups, idx_gray, ds