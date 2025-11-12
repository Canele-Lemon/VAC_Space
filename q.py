def build_gray_table(sparse_high_csv: str, low_4096_csv: str) -> pd.DataFrame:
    # 희소 High 로드
    df_s = pd.read_csv(sparse_high_csv)
    _ensure_sparse_columns(df_s)
    _coerce_sparse_types(df_s)
    _apply_gray254_rule(df_s)

    # Gray, LUT_j NaN 제거 및 정렬(같은 Gray가 여러 번이면 마지막 행 유지)
    df_s = df_s.dropna(subset=["Gray", "LUT_j"]).copy()
    df_s["Gray"] = df_s["Gray"].astype(int)
    df_s = df_s.sort_values(["Gray"], kind="mergesort")
    df_s = df_s.groupby("Gray", as_index=False).tail(1)  # 같은 gray는 마지막 것

    # -------- High: LUT_j축 0..4095로 선형 보간 --------
    # (여기서 j=16과 j=36 사이 17..35의 High(R/G/B) 값이 채워집니다.)
    df_anchor = _collapse_duplicate_j_keep_last(
        df_s[["LUT_j", "R_High", "G_High", "B_High"]], gray_col=None
    )
    df_high4096 = _interp_to_4096(df_anchor).set_index("LUT_j")

    # -------- Low: 12bit 원본 4096 그대로 --------
    df_low4096 = _load_low_4096(low_4096_csv).set_index("LUT_j")

    # -------- Gray→j 매핑 자체도 선형보간으로 ‘조밀화’ --------
    # 예) Gray=2→16, Gray=3→36이면, 전체 Gray(0..255)에 대해 j(g)를 선형 보간
    gray_anchor = df_s["Gray"].to_numpy(dtype=np.float64)
    j_anchor    = df_s["LUT_j"].to_numpy(dtype=np.float64)

    # 앵커가 2개 미만이면 보간 불가
    if gray_anchor.size < 2:
        raise ValueError("Gray→LUT_j 보간을 위해서는 서로 다른 Gray 앵커가 최소 2개 필요합니다.")

    full_gray = np.arange(256, dtype=np.float64)
    lut_j_dense = np.interp(full_gray, gray_anchor, j_anchor)  # Gray축 보간된 j(g)

    # 정수화 + 경계 클립
    lut_j_dense = np.rint(lut_j_dense).astype(np.int32)
    lut_j_dense = np.clip(lut_j_dense, 0, 4095)

    # 단조 증가(비감소) 강제: 이전보다 작아지면 누적 max로 올림
    lut_j_dense = np.maximum.accumulate(lut_j_dense)

    # 규칙 강제: gray254=4092, gray255=4092
    lut_j_dense[254] = 4092
    lut_j_dense[255] = 4092
    # (단조성 유지 위해 0..254도 4092를 넘지 않도록)
    lut_j_dense[:255] = np.minimum(lut_j_dense[:255], 4092)

    # -------- 256행 테이블 생성(보간된 j(g) 사용) --------
    rows = []
    for g in range(256):
        j = int(lut_j_dense[g])

        # Low는 원본에서 j 그대로
        R_low = df_low4096.at[j, "R_Low_full"]
        G_low = df_low4096.at[j, "G_Low_full"]
        B_low = df_low4096.at[j, "B_Low_full"]

        # High는 보간된 4096에서 j 그대로
        R_hih = df_high4096.at[j, "R_High_full"]
        G_hih = df_high4096.at[j, "G_High_full"]
        B_hih = df_high4096.at[j, "B_High_full"]

        rows.append({
            "GrayLevel_window": g,
            "R_Low":  int(round(R_low)),
            "R_High": int(round(R_hih)),
            "G_Low":  int(round(G_low)),
            "G_High": int(round(G_hih)),
            "B_Low":  int(round(B_low)),
            "B_High": int(round(B_hih)),
        })

    df_out256 = pd.DataFrame(rows)
    return df_out256