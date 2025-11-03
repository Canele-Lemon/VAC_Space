def build_white_X_Y3(pk_list, ref_pk):
    """
    세 컴포넌트(dCx, dCy, dGamma)의 (pk, gray) 교집합만 남겨
    X(공통), Y3=[dCx,dCy,dGamma]를 같은 순서로 정렬해서 반환.
    """
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_pk)

    X_cx, y_cx, g_cx = ds.build_white_y0_delta(component='dCx')
    X_cy, y_cy, g_cy = ds.build_white_y0_delta(component='dCy')
    X_ga, y_ga, g_ga = ds.build_white_y0_delta(component='dGamma')

    # 공통: gray_norm 위치 계산
    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])
    idx_gray_cx = 3 + K + 2
    idx_gray_cy = 3 + K + 2
    idx_gray_ga = 3 + K + 2

    def keys_from(X, groups, idx_gray):
        gray_norm = X[:, idx_gray]
        gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)
        # 키 = (pk, gray)
        return np.stack([groups.astype(np.int64), gray_idx.astype(np.int64)], axis=1)

    keys_cx = keys_from(X_cx, g_cx, idx_gray_cx)
    keys_cy = keys_from(X_cy, g_cy, idx_gray_cy)
    keys_ga = keys_from(X_ga, g_ga, idx_gray_ga)

    # 각 키를 문자열로 만들어 집합 교집합
    def key_str(keys):
        return np.char.add(keys[:,0].astype(str), ":" + keys[:,1].astype(str))

    kc = key_str(keys_cx)
    ky = key_str(keys_cy)
    kg = key_str(keys_ga)

    common = np.intersect1d(np.intersect1d(kc, ky), kg)
    if common.size == 0:
        raise RuntimeError("세 컴포넌트의 (pk,gray) 교집합이 비어 있습니다.")

    # 공통 키의 인덱스 선택자를 만든다
    def indexer(all_keys_str, common_keys):
        # 공통 키 순서에 맞춰 인덱스 배열 생성
        lookup = {k:i for i,k in enumerate(all_keys_str)}
        return np.array([lookup[k] for k in common_keys], dtype=np.int64)

    idx_cx = indexer(kc, common)
    idx_cy = indexer(ky, common)
    idx_ga = indexer(kg, common)

    # 동일 행 순서로 정렬
    X = X_cx[idx_cx].astype(np.float32)          # X는 어떤 컴포넌트에서 가져와도 동일
    y_cx_sel = y_cx[idx_cx].astype(np.float32)
    y_cy_sel = y_cy[idx_cy].astype(np.float32)
    y_ga_sel = y_ga[idx_ga].astype(np.float32)

    # g=2..253만 사용 + 세 타깃 모두 finite
    gray_norm = X[:, 3 + K + 2]
    gray_idx  = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)
    Y3 = np.stack([y_cx_sel, y_cy_sel, y_ga_sel], axis=1)

    core_mask = (gray_idx >= 2) & (gray_idx <= 253) & np.isfinite(Y3).all(axis=1)
    X   = X[core_mask]
    Y3  = Y3[core_mask]
    # groups는 pk, gray에서 pk만 필요하면 pk만 복원
    # pk는 common 키에서 앞부분이므로 재구성
    groups = np.array([int(k.split(":")[0]) for k in common], dtype=np.int64)[core_mask]

    # gray_norm 컬럼 인덱스 반환 (나중에 사용할 수 있도록)
    idx_gray = 3 + K + 2
    return X, Y3, groups, idx_gray, ds