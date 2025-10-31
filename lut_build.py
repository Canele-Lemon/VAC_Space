# lut_build_high_from_knots_csv_keep_low_gray.py
# - Low: 기존 CSV에서 그대로 사용 (4096포인트)
# - High: HIGH_KNOT_CSV의 (Gray8, Gray12, R_High, G_High, B_High) 34점 → 선형 보간
# - 첫 열: GrayLevel_window (0~4095)
# - 출력 열: GrayLevel_window, R_Low, R_High, G_Low, G_High, B_Low, B_High

import numpy as np
import pandas as pd
import os

# =======================
# 경로/설정
# =======================
INPUT_LOW_CSV   = r"./your_low_lut_4096.csv"          # 기존 Low 채널 CSV (열: R_Low,G_Low,B_Low)
HIGH_KNOT_CSV   = r"./high_knots_values.csv"          # (Gray8, Gray12, R_High, G_High, B_High)
OUTPUT_PATH     = r"./LUT_full_4096_high_from_knots_with_gray.csv"

FULL_POINTS = 4096
EXPECTED_COUNT = 34           # 34개 제어점
ENFORCE_MONOTONE = True
EPS_HIGH_OVER_LOW = 1         # High ≥ Low + EPS

LOW_COLS = ["R_Low", "G_Low", "B_Low"]  # INPUT_LOW_CSV에 필요한 열

# 엣지 값 고정: 1번=0, 33·34번=4095  (0-based index로 0, 32, 33)
FORCE_FIRST_ZERO_IDX = 0
FORCE_4095_IDXS = [32, 33]

# =======================
# 유틸
# =======================
def _enforce_monotone(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float).copy()
    for i in range(1, a.size):
        if a[i] < a[i-1]:
            a[i] = a[i-1]
    return a

def _clip_round_12bit(arr: np.ndarray) -> np.ndarray:
    a = np.clip(np.rint(arr), 0, FULL_POINTS - 1)
    return a.astype(np.uint16)

def _validate_low_csv(df_low: pd.DataFrame):
    for c in LOW_COLS:
        if c not in df_low.columns:
            raise ValueError(f"입력 Low CSV에 '{c}' 열이 필요합니다.")
    if len(df_low) != FULL_POINTS:
        raise ValueError(f"입력 Low CSV 행 수={len(df_low)} (4096이어야 합니다).")

def _validate_knots(gray12: np.ndarray, gray8: np.ndarray):
    if gray12.ndim != 1 or gray8.ndim != 1:
        raise ValueError("Gray8, Gray12는 1차원 벡터여야 합니다.")
    if len(gray12) != EXPECTED_COUNT or len(gray8) != EXPECTED_COUNT:
        raise ValueError(f"CSV 제어점 개수={len(gray12)} (예상 {EXPECTED_COUNT})")
    if np.any(gray12 < 0) or np.any(gray12 > FULL_POINTS-1):
        raise ValueError("Gray12 값은 0~4095 범위여야 합니다.")
    if np.any(gray8 < 0) or np.any(gray8 > 255):
        raise ValueError("Gray8 값은 0~255 범위여야 합니다.")
    if not np.all(np.diff(gray12) >= 0):
        raise ValueError("Gray12는 오름차순(단조 비감소)이어야 합니다.")
    if gray12[0] != 0:
        print("[WARN] 첫 Gray12가 0이 아닙니다. 외삽(extrapolation) 대신 0을 포함하는 것을 권장합니다.")
    if gray12[-1] != 4095:
        print("[WARN] 마지막 Gray12가 4095가 아닙니다. 외삽(extrapolation) 대신 4095를 포함하는 것을 권장합니다.")

def _apply_edge_value_rules(r_vals, g_vals, b_vals):
    """값(High) 측면에서 엣지 규칙 강제: 첫=0, 33·34=4095"""
    def fix(v):
        v = np.asarray(v, dtype=float).copy()
        # 첫 knot 값 = 0
        if 0 <= FORCE_FIRST_ZERO_IDX < v.size:
            v[FORCE_FIRST_ZERO_IDX] = 0.0
        # 33·34번째 knot 값 = 4095
        for idx in FORCE_4095_IDXS:
            if 0 <= idx < v.size:
                v[idx] = 4095.0
        return np.clip(v, 0, FULL_POINTS-1)
    return fix(r_vals), fix(g_vals), fix(b_vals)

def _interp_to_4096(gray12_knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    x_big = np.arange(FULL_POINTS, dtype=float)
    # numpy.interp는 x가 엄밀히 증가해야 하는데, 동일값이 있을 수 있으면 약간의 jiggle 방지:
    # (여기서는 이미 단조 비감소 검증했고, 동일값은 허용하지만 구간 길이가 0이면 해당 y가 평평하게 유지됨)
    return np.interp(x_big, gray12_knots.astype(float), values.astype(float))

# =======================
# 메인
# =======================
def main():
    # 1) Low 채널 읽기/검증
    df_low = pd.read_csv(INPUT_LOW_CSV)
    _validate_low_csv(df_low)
    R_low = df_low["R_Low"].to_numpy(float)
    G_low = df_low["G_Low"].to_numpy(float)
    B_low = df_low["B_Low"].to_numpy(float)
    if ENFORCE_MONOTONE:
        R_low = _enforce_monotone(R_low)
        G_low = _enforce_monotone(G_low)
        B_low = _enforce_monotone(B_low)

    # 2) High KNOT CSV 읽기
    #    요구 열: Gray8, Gray12, R_High, G_High, B_High
    df_k = pd.read_csv(HIGH_KNOT_CSV)
    required_cols = ["Gray8", "Gray12", "R_High", "G_High", "B_High"]
    for c in required_cols:
        if c not in df_k.columns:
            raise ValueError(f"HIGH_KNOT_CSV에 '{c}' 열이 필요합니다.")

    # 정렬(혹시 무순서 입력 시)
    df_k = df_k.sort_values(["Gray12", "Gray8"]).reset_index(drop=True)

    gray8  = df_k["Gray8"].to_numpy(int)
    gray12 = df_k["Gray12"].to_numpy(int)
    Rv     = df_k["R_High"].to_numpy(float)
    Gv     = df_k["G_High"].to_numpy(float)
    Bv     = df_k["B_High"].to_numpy(float)

    # 3) KNOTS/값 검증 + 엣지 값 규칙 적용
    _validate_knots(gray12, gray8)
    Rv, Gv, Bv = _apply_edge_value_rules(Rv, Gv, Bv)

    # 4) 선형 보간 → 4096 포인트
    R_high = _interp_to_4096(gray12, Rv)
    G_high = _interp_to_4096(gray12, Gv)
    B_high = _interp_to_4096(gray12, Bv)

    if ENFORCE_MONOTONE:
        R_high = _enforce_monotone(R_high)
        G_high = _enforce_monotone(G_high)
        B_high = _enforce_monotone(B_high)

    # High ≥ Low + EPS
    R_high = np.maximum(R_high, R_low + EPS_HIGH_OVER_LOW)
    G_high = np.maximum(G_high, G_low + EPS_HIGH_OVER_LOW)
    B_high = np.maximum(B_high, B_low + EPS_HIGH_OVER_LOW)

    # 5) 출력 DataFrame
    out_df = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":  _clip_round_12bit(R_low),
        "R_High": _clip_round_12bit(R_high),
        "G_Low":  _clip_round_12bit(G_low),
        "G_High": _clip_round_12bit(G_high),
        "B_Low":  _clip_round_12bit(B_low),
        "B_High": _clip_round_12bit(B_high),
    })

    # 6) 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Saved: {os.path.abspath(OUTPUT_PATH)}")
    print(f" - #knots: {len(gray12)} (expected {EXPECTED_COUNT})")
    print(f" - rows  : {len(out_df)} (expected {FULL_POINTS})")

if __name__ == "__main__":
    main()