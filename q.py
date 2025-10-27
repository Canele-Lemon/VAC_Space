def debug_preview_training_rows(pk_list=[2444], component='dGamma', patterns=('W',), max_print=5):
    """
    pk_list에 있는 panel들로 VACDataset을 만들고
    build_per_gray_y0() 결과 중 앞 max_print개 행을 사람이 읽을 수 있게 출력.

    component: 'dGamma' | 'dCx' | 'dCy'
    patterns : ('W',) 등으로 제한 가능
    """
    ds = VACDataset(pk_list=pk_list)

    X_mat, y_vec, groups = ds.build_per_gray_y0(component=component, patterns=patterns)

    print("[DEBUG] build_per_gray_y0 result")
    print("  X_mat shape:", X_mat.shape)
    print("  y_vec shape:", y_vec.shape)
    print("  groups shape:", groups.shape)

    # panel onehot 길이
    panel_len = len(ds.samples[0]["X"]["meta"]["panel_maker"])

    for i in range(min(max_print, X_mat.shape[0])):
        feat = X_mat[i]
        y    = y_vec[i]
        pk   = groups[i]

        # feat layout 복원
        # 0:6 = ΔR_Low, ΔR_High, ΔG_Low, ΔG_High, ΔB_Low, ΔB_High
        delta_lut_6 = feat[:6]

        # panel onehot 다음 위치들 계산
        idx_panel_start = 6
        idx_panel_end   = 6 + panel_len
        panel_oh        = feat[idx_panel_start:idx_panel_end]

        frame_rate      = feat[idx_panel_end + 0]
        model_year      = feat[idx_panel_end + 1]
        gray_norm       = feat[idx_panel_end + 2]

        pattern_onehot  = feat[idx_panel_end + 3 : idx_panel_end + 7]

        # gray 추정도 해볼 수 있음: gray_norm ~= g/255
        est_gray = gray_norm * 255.0

        # pattern 추정: argmax of pattern_onehot
        ptn_idx = int(np.argmax(pattern_onehot))
        ptn_map = ['W','R','G','B']
        ptn     = ptn_map[ptn_idx]

        print(f"\n--- sample {i} ---")
        print(f"pk: {pk}")
        print(f"pattern_onehot -> {pattern_onehot}  => pattern '{ptn}'")
        print(f"gray_norm      -> {gray_norm:.6f}  (≈ gray {est_gray:.1f})")
        print(f"ΔLUT[0:6]      -> [ΔR_Low, ΔR_High, ΔG_Low, ΔG_High, ΔB_Low, ΔB_High]")
        print(f"                  {delta_lut_6}")
        print(f"panel_onehot   -> {panel_oh}")
        print(f"frame_rate     -> {frame_rate}")
        print(f"model_year     -> {model_year}")
        print(f"target {component} (y) -> {y}")