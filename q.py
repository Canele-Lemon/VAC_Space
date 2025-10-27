def build_per_gray_y0(self, component='dGamma', patterns=('W','R','G','B')):
    """
    자코비안/보정 학습용 1D 회귀 데이터셋 생성.

    각 row는 (pk, pattern p, gray g)에 해당.
    X_row 는 ΔLUT 기반 피처 (prepare_X_delta() 결과에서 나온 lut)
    y_val 는 Δ응답 (dGamma / dCx / dCy), 즉 target - ref

    Parameters
    ----------
    component : {'dGamma','dCx','dCy'}
    patterns  : tuple of patterns to include ('W','R','G','B')

    Returns
    -------
    X_mat : (N, D)
    y_vec : (N,)
    groups: (N,) pk ID for each row (useful for grouped CV 등)
    """
    assert component in ('dGamma','dCx','dCy')

    X_rows, y_vals, groups = [], [], []

    for s in self.samples:
        pk  = s["pk"]
        Xd  = s["X"]  # this is now ΔLUT dict (prepare_X_delta)
        Yd  = s["Y"]  # this is now ΔY dict (prepare_Y -> compute_Y0_struct)

        for p in patterns:
            y_vec = Yd['Y0'][p][component]  # (256,)
            for g in range(256):
                y_val = y_vec[g]
                if not np.isfinite(y_val):
                    continue

                feat_row = self._build_features_for_gray(
                    X_dict=Xd,
                    gray=g,
                    add_pattern=p
                )

                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

    if X_rows:
        X_mat = np.vstack(X_rows).astype(np.float32)
    else:
        X_mat = np.empty((0,0), dtype=np.float32)

    y_vec = np.asarray(y_vals, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)

    return X_mat, y_vec, groups