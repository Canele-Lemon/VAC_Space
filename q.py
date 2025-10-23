if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)

    target_pk = 500
    dataset = VACDataset([target_pk])

    # ─────────────────────────────────────────────────────
    # 1) per-gray Y0('W' 패턴, 예: Cx) 학습셋을 구성해서 구조 확인
    #    (훈련 때 single-target 회귀 입력 스키마 확인용)
    # ─────────────────────────────────────────────────────
    X_mat, y_vec, groups = dataset.build_per_gray_y0(component='Cx', patterns=('W',))

    print("\n[DEBUG] build_per_gray_y0 (component='Cx', pattern='W')")
    print(f"  X_mat shape: {X_mat.shape}")      # (N_rows, D)
    print(f"  y_vec shape: {y_vec.shape}")      # (N_rows,)
    print(f"  groups shape: {groups.shape}")    # (N_rows,)
    print(f"  unique pk in groups: {np.unique(groups)}")

    # 피처 스키마 요약: (앞 6=6채널 LUT, 이후 panel_onehot + 2(meta) + 1(gray_norm) + 4(pattern_onehot))
    # panel_onehot 길이 유추
    D = X_mat.shape[1]
    # 뒤에서 패턴 원핫 4개가 항상 마지막 4개라는 가정(prepare_X0/feature 함수 기준)
    pattern_oh = X_mat[-1, -4:]
    # panel_onehot 길이는 D - (6 LUT + 2 meta + 1 gray + 4 pattern)
    inferred_panel_len = D - (6 + 2 + 1 + 4)
    print(f"  inferred panel_onehot length: {inferred_panel_len}")
    print(f"  tail pattern_onehot sample(last row): {pattern_oh}")

    # 샘플 행 몇 개 덤프 (중간과 끝부분)
    def _fmt_row(v):
        head = ", ".join(f"{x:.4f}" for x in v[:10])
        tail = ", ".join(f"{x:.4f}" for x in v[-10:])
        return f"[{head}, ..., {tail}] (len={len(v)})"

    mid_idx = len(X_mat)//2
    print(f"\n  sample row @idx=0   : {_fmt_row(X_mat[0])}, y={y_vec[0]:.6f}, pk={groups[0]}")
    print(f"  sample row @idx={mid_idx}: {_fmt_row(X_mat[mid_idx])}, y={y_vec[mid_idx]:.6f}, pk={groups[mid_idx]}")
    print(f"  sample row @idx=-1  : {_fmt_row(X_mat[-1])}, y={y_vec[-1]:.6f}, pk={groups[-1]}")

    # ─────────────────────────────────────────────────────
    # 2) “뒤에서 10개까지”의 데이터(행) 출력
    # ─────────────────────────────────────────────────────
    tail_n = min(10, X_mat.shape[0])
    print(f"\n[DEBUG] Last {tail_n} rows (features tail & y):")
    for i in range(tail_n):
        idx = X_mat.shape[0] - tail_n + i
        x_row = X_mat[idx]
        y_val = y_vec[idx]
        grp   = groups[idx]
        print(f"  idx={idx:>6} | y={y_val:.6f} | pk={grp} | tail10=({', '.join(f'{v:.4f}' for v in x_row[-10:])})")

    # ─────────────────────────────────────────────────────
    # 3) 멀티타깃 플랫 스키마도 한 번 체크(선택)
    # ─────────────────────────────────────────────────────
    X_flat, Y_flat = dataset.build_multitarget_flat(include=('Y0','Y1','Y2'))
    print("\n[DEBUG] build_multitarget_flat(include=Y0,Y1,Y2)")
    print(f"  X_flat shape: {X_flat.shape}  # 6*256 + |panel_onehot| + 2")
    print(f"  Y_flat shape: {Y_flat.shape}  # 3072 + 255 + 12 per PK")