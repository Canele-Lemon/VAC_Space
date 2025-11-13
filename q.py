def _build_single_lut(R_OFFSET, G_OFFSET, B_OFFSET, base_name=None):
    # 1) Low LUT (12bit, 4096포인트) 로드
    df_low = pd.read_csv(LOW_LUT_CSV)
    Rl = df_low["R_Low"].to_numpy(float)
    Gl = df_low["G_Low"].to_numpy(float)
    Bl = df_low["B_Low"].to_numpy(float)
    Rl, Gl, Bl = _enforce_monotone(Rl), _enforce_monotone(Gl), _enforce_monotone(Bl)

    # 2) High knot(256포인트) 로드
    dfk    = pd.read_csv(HIGH_KNOT_CSV)
    gray8  = dfk["Gray8"].to_numpy(int)        # 0~255
    gray12 = dfk["Gray12"].to_numpy(float)     # 256개, 0~4095 중 일부
    Rk     = dfk["R_High"].to_numpy(float)
    Gk     = dfk["G_High"].to_numpy(float)
    Bk     = dfk["B_High"].to_numpy(float)

    # 3) 전체 knot에 offset 적용 (여기서는 특별히 lock 안 걸고, 나중에 12bit에서 lock)
    Rk = Rk + R_OFFSET
    Gk = Gk + G_OFFSET
    Bk = Bk + B_OFFSET

    # 4) Low + eps 제약 (knot domain)
    Rk = _enforce_low_eps(gray12, Rk, Rl, EPS_HIGH_OVER_LOW)
    Gk = _enforce_low_eps(gray12, Gk, Gl, EPS_HIGH_OVER_LOW)
    Bk = _enforce_low_eps(gray12, Bk, Bl, EPS_HIGH_OVER_LOW)

    # 5) 단조 증가 강제 (knot domain)
    Rk = _enforce_monotone(Rk)
    Gk = _enforce_monotone(Gk)
    Bk = _enforce_monotone(Bk)

    # 6) 256 knot → 4096 LUT (12bit) 보간
    Rh = np.clip(_interp_knots_to_4096(gray12, Rk), 0, 4095)
    Gh = np.clip(_interp_knots_to_4096(gray12, Gk), 0, 4095)
    Bh = np.clip(_interp_knots_to_4096(gray12, Bk), 0, 4095)

    # ─────────────────────────────
    # 7) 12bit index 기준 LOCK 적용
    #    중요: Low+eps 제약이 이 인덱스 값들을 밀어올리지 못하게 마스크 처리
    # ─────────────────────────────
    LOCK_VALS = {
        0:    0.0,    # Gray8=0,1 → Gray12=0
        4092: 4092.0, # Gray8=254 → Gray12=4092
        4095: 4095.0, # Gray8=255 → Gray12=4095
    }
    lock_idx = np.array(list(LOCK_VALS.keys()), dtype=int)

    # 7-1) Low+eps 제약을 non-locked 구간에만 적용
    mask = np.ones_like(Rh, dtype=bool)
    mask[lock_idx] = False

    Rh[mask] = np.maximum(Rh[mask], Rl[mask] + EPS_HIGH_OVER_LOW)
    Gh[mask] = np.maximum(Gh[mask], Gl[mask] + EPS_HIGH_OVER_LOW)
    Bh[mask] = np.maximum(Bh[mask], Bl[mask] + EPS_HIGH_OVER_LOW)

    # 7-2) locked index에 원하는 값 강제 세팅
    for j, v in LOCK_VALS.items():
        if 0 <= j < FULL_POINTS:
            Rh[j] = v
            Gh[j] = v
            Bh[j] = v

    # 8) (선택) 전체 4096 도메인에서 단조 확인용으로 다시 한 번 monotone을 걸 수도 있지만
    #    그러면 잠금 값이 또 밀릴 수 있으니 여기서는 생략하거나,
    #    monotone 후에 다시 LOCK_VALS 덮어써도 됩니다.
    #    ex)
    # Rh = _enforce_monotone(Rh); Gh = _enforce_monotone(Gh); Bh = _enforce_monotone(Bh)
    # for j, v in LOCK_VALS.items():
    #     Rh[j] = v; Gh[j] = v; Bh[j] = v

    # 9) 최종 DataFrame 구성
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