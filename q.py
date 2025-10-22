# VAC_dataset.py (build_per_gray_y0) - NaN을 드롭
def build_per_gray_y0(self, component='Gamma', patterns=('W','R','G','B')):
    assert component in ('Gamma', 'Cx', 'Cy')
    X_rows, y_vals = [], []
    for s in self.samples:
        Xd = s["X"]; Yd = s["Y"]
        for p in patterns:
            y_vec = Yd['Y0'][p][component]  # (256,)
            for g in range(256):
                y_val = y_vec[g]
                if not np.isfinite(y_val):     # NaN/inf는 스킵
                    continue
                X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
                y_vals.append(float(y_val))
    X_mat = np.vstack(X_rows).astype(np.float32)
    y_vec = np.asarray(y_vals, dtype=np.float32)
    return X_mat, y_vec