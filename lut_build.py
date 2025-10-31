# lut_build_high_from_33_keep_low_gray.py
# - Low: 기존 CSV에서 그대로 사용 (4096포인트)
# - High: 공통 33포인트 → 선형 보간
# - 첫 열: GrayLevel_window (0~4095)
# - 출력 열 순서: GrayLevel_window, R_Low,R_High,G_Low,G_High,B_Low,B_High

import numpy as np
import pandas as pd
import os

# =======================
# 경로/설정
# =======================
INPUT_LOW_CSV  = r"./your_low_lut_4096.csv"   # 기존 Low 채널 CSV 경로
OUTPUT_PATH    = r"./LUT_full_4096_high_from_33_with_gray.csv"

FULL_POINTS = 4096
KNOT_COUNT  = 33
ENFORCE_MONOTONE = True
EPS_HIGH_OVER_LOW = 1

LOW_COLS = ["R_Low", "G_Low", "B_Low"]        # 입력 CSV에 있어야 하는 열 이름

# =======================
# 1) 33포인트 인덱스 (공통)
# =======================
# KNOTS = np.linspace(0, FULL_POINTS - 1, KNOT_COUNT, dtype=int).tolist()
KNOTS = [0,    # KNOT 1
         112,  # KNOT 2
         240,  # KNOT 3
         368,  # KNOT 4
         499,  # KNOT 5
         626,  # KNOT 6
         758,  # KNOT 7
         889,  # KNOT 8
         1020, # KNOT 9
         1148, # KNOT 10
         1279, # KNOT 11
         1410, # KNOT 12
         1522, # KNOT 13
         1653, # KNOT 14
         1780, # KNOT 15
         1911, # KNOT 16
         2043, # KNOT 17
         2175, # KNOT 18
         2304, # KNOT 19
         2437, # KNOT 20
         2570, # KNOT 21
         2699, # KNOT 22
         2812, # KNOT 23
         2945, # KNOT 24
         3078, # KNOT 25
         3211, # KNOT 26
         3340, # KNOT 27
         3473, # KNOT 28
         3606, # KNOT 29
         3739, # KNOT 30
         3868, # KNOT 31
         4000, # KNOT 32
         4092, # KNOT 33
         4095  # KNOT 34
         ]

# =======================
# 2) High 채널용 33값 지정 (길이=33)
# =======================
def _auto_baseline_values(offset=1):
    x_small = np.array(KNOTS, dtype=float)
    y_small = x_small + offset
    return np.clip(y_small, 0, FULL_POINTS - 1).tolist()

CHANNELS = {
    "R_High": _auto_baseline_values(offset=1),
    "G_High": _auto_baseline_values(offset=1),
    "B_High": _auto_baseline_values(offset=1),
}

# =======================
# 유틸
# =======================
def _interp_33_to_4096(knots, values):
    x_big = np.arange(FULL_POINTS, dtype=float)
    return np.interp(x_big, np.array(knots, float), np.array(values, float))

def _enforce_monotone(arr):
    a = np.asarray(arr, dtype=float).copy()
    for i in range(1, a.size):
        if a[i] < a[i-1]:
            a[i] = a[i-1]
    return a

def _clip_round_12bit(arr):
    a = np.clip(np.rint(arr), 0, FULL_POINTS - 1)
    return a.astype(np.uint16)

def _validate_inputs(knots, channels_dict):
    if len(knots) != KNOT_COUNT:
        raise ValueError(f"KNOTS 길이={len(knots)} (예상 {KNOT_COUNT})")
    if list(sorted(knots)) != list(knots):
        raise ValueError("KNOTS는 오름차순이어야 합니다.")
    for k in ["R_High","G_High","B_High"]:
        if k not in channels_dict:
            raise ValueError(f"CHANNELS에 '{k}'가 필요합니다.")
        if len(channels_dict[k]) != len(knots):
            raise ValueError(f"{k} 길이({len(channels_dict[k])}) != KNOTS 길이({len(knots)})")

# =======================
# 메인
# =======================
def main():
    _validate_inputs(KNOTS, CHANNELS)

    # 1) Low 채널 읽기
    df_low = pd.read_csv(INPUT_LOW_CSV)
    for c in LOW_COLS:
        if c not in df_low.columns:
            raise ValueError(f"입력 CSV에 '{c}' 열이 필요합니다.")
    if len(df_low) != FULL_POINTS:
        raise ValueError(f"입력 CSV 행 수={len(df_low)} (4096이어야 합니다).")

    R_low = np.asarray(df_low["R_Low"], dtype=float)
    G_low = np.asarray(df_low["G_Low"], dtype=float)
    B_low = np.asarray(df_low["B_Low"], dtype=float)

    if ENFORCE_MONOTONE:
        R_low = _enforce_monotone(R_low)
        G_low = _enforce_monotone(G_low)
        B_low = _enforce_monotone(B_low)

    # 2) High 채널 보간
    R_high = _interp_33_to_4096(KNOTS, CHANNELS["R_High"])
    G_high = _interp_33_to_4096(KNOTS, CHANNELS["G_High"])
    B_high = _interp_33_to_4096(KNOTS, CHANNELS["B_High"])

    if ENFORCE_MONOTONE:
        R_high = _enforce_monotone(R_high)
        G_high = _enforce_monotone(G_high)
        B_high = _enforce_monotone(B_high)

    # High >= Low + EPS
    R_high = np.maximum(R_high, R_low + EPS_HIGH_OVER_LOW)
    G_high = np.maximum(G_high, G_low + EPS_HIGH_OVER_LOW)
    B_high = np.maximum(B_high, B_low + EPS_HIGH_OVER_LOW)

    # 3) 결과 DataFrame
    out_df = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":  _clip_round_12bit(R_low),
        "R_High": _clip_round_12bit(R_high),
        "G_Low":  _clip_round_12bit(G_low),
        "G_High": _clip_round_12bit(G_high),
        "B_Low":  _clip_round_12bit(B_low),
        "B_High": _clip_round_12bit(B_high),
    })

    # 4) 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Saved: {os.path.abspath(OUTPUT_PATH)}")
    print(f" - KNOTS size : {len(KNOTS)}")
    print(f" - Output rows: {len(out_df)} (should be {FULL_POINTS})")

if __name__ == "__main__":
    main()
