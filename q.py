def _build_single_lut(R_OFFSET, G_OFFSET, B_OFFSET, base_name=None):
    # 1) Low 4096 로드
    df_low = pd.read_csv(LOW_LUT_CSV)
    Rl = df_low["R_Low"].to_numpy(float)
    Gl = df_low["G_Low"].to_numpy(float)
    Bl = df_low["B_Low"].to_numpy(float)
    Rl, Gl, Bl = _enforce_monotone(Rl), _enforce_monotone(Gl), _enforce_monotone(Bl)

    # 2) High knot(Gray8, Gray12, R_High, G_High, B_High) 로드
    dfk   = pd.read_csv(HIGH_KNOT_CSV)
    gray8 = dfk["Gray8"].to_numpy(int)
    gray12 = dfk["Gray12"].to_numpy(float)
    Rk = dfk["R_High"].to_numpy(float)
    Gk = dfk["G_High"].to_numpy(float)
    Bk = dfk["B_High"].to_numpy(float)

    # 2-1) knot에서 Gray8 기준 잠금 값 세팅
    # Gray8=0,1 → 0 / Gray8=254 → 4092 / Gray8=255 → 4095
    FIXED_KNOT = {}
    for idx, g in enumerate(gray8):
        if g in (0, 1):
            FIXED_KNOT[idx] = 0.0
        elif g == 254:
            FIXED_KNOT[idx] = 4092.0
        elif g == 255:
            FIXED_KNOT[idx] = 4095.0
    LOCKED_KNOT_IDX = set(FIXED_KNOT.keys())

    # 먼저 한 번 고정값을 세팅
    for idx, v in FIXED_KNOT.items():
        Rk[idx] = v
        Gk[idx] = v
        Bk[idx] = v

    # 3) 전체 knot에 offset 적용 (단, 잠금 knot는 제외)
    for i in range(len(Rk)):
        if i not in LOCKED_KNOT_IDX:
            Rk[i] += R_OFFSET
            Gk[i] += G_OFFSET
            Bk[i] += B_OFFSET

    # 다시 잠금값 덮어쓰기 (숫자 오차 방지)
    for idx, v in FIXED_KNOT.items():
        Rk[idx] = v
        Gk[idx] = v
        Bk[idx] = v

    # 4) Low + eps 제약 (knot 도메인) – 여기서는 전 구간에 적용해도 되고,
    #    적용 후 다시 FIXED_KNOT을 덮어써도 OK
    Rk = _enforce_low_eps(gray12, Rk, Rl, EPS_HIGH_OVER_LOW)
    Gk = _enforce_low_eps(gray12, Gk, Gl, EPS_HIGH_OVER_LOW)
    Bk = _enforce_low_eps(gray12, Bk, Bl, EPS_HIGH_OVER_LOW)

    for idx, v in FIXED_KNOT.items():
        Rk[idx] = v
        Gk[idx] = v
        Bk[idx] = v

    # 5) 단조 증가 강제 (knot 도메인)
    Rk = _enforce_monotone(Rk)
    Gk = _enforce_monotone(Gk)
    Bk = _enforce_monotone(Bk)

    # 다시 한 번 잠금값 덮어쓰기
    for idx, v in FIXED_KNOT.items():
        Rk[idx] = v
        Gk[idx] = v
        Bk[idx] = v

    # 6) 256 knot → 4096 LUT 보간
    Rh = np.clip(_interp_knots_to_4096(gray12, Rk), 0, 4095)
    Gh = np.clip(_interp_knots_to_4096(gray12, Gk), 0, 4095)
    Bh = np.clip(_interp_knots_to_4096(gray12, Bk), 0, 4095)

    # 7) 12bit 도메인에서 잠금 + low+eps + monotone 처리
    Rh = _enforce_with_locks_12bit(Rh, Rl, EPS_HIGH_OVER_LOW)
    Gh = _enforce_with_locks_12bit(Gh, Gl, EPS_HIGH_OVER_LOW)
    Bh = _enforce_with_locks_12bit(Bh, Bl, EPS_HIGH_OVER_LOW)

    # 8) 최종 DataFrame 구성
    out = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":  _clip_round_12bit(Rl),
        "R_High": _clip_round_12bit(Rh),
        "G_Low":  _clip_round_12bit(Gl),
        "G_High": _clip_round_12bit(Gh),
        "B_Low":  _clip_round_12bit(Bl),
        "B_High": _clip_round_12bit(Bh),
    })
    return out