# lut_build_high_from_knots_csv_keep_low_gray_plot.py
# - Low: 기존 CSV에서 그대로 사용 (4096포인트)
# - High: HIGH_KNOT_CSV의 (Gray8, Gray12, R_High, G_High, B_High) 34점 → 선형 보간
# - 첫 열: GrayLevel_window (0~4095)
# - 출력 열: GrayLevel_window, R_Low,R_High,G_Low,G_High,B_Low,B_High
# - 그래프: 34포인트 vs 보간 결과 비교 표시

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# =======================
# 경로/설정
# =======================
INPUT_LOW_CSV   = r"./your_low_lut_4096.csv"
HIGH_KNOT_CSV   = r"./high_knots_values.csv"
OUTPUT_PATH     = r"./LUT_full_4096_high_from_knots_with_gray.csv"

FULL_POINTS = 4096
EXPECTED_COUNT = 34
ENFORCE_MONOTONE = True
EPS_HIGH_OVER_LOW = 1

LOW_COLS = ["R_Low", "G_Low", "B_Low"]
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
    if len(gray12) != EXPECTED_COUNT:
        raise ValueError(f"CSV 제어점 개수={len(gray12)} (예상 {EXPECTED_COUNT})")
    if not np.all(np.diff(gray12) >= 0):
        raise ValueError("Gray12는 오름차순이어야 합니다.")

def _apply_edge_value_rules(r_vals, g_vals, b_vals):
    def fix(v):
        v = np.asarray(v, dtype=float).copy()
        if 0 <= FORCE_FIRST_ZERO_IDX < v.size:
            v[FORCE_FIRST_ZERO_IDX] = 0.0
        for idx in FORCE_4095_IDXS:
            if 0 <= idx < v.size:
                v[idx] = 4095.0
        return np.clip(v, 0, FULL_POINTS-1)
    return fix(r_vals), fix(g_vals), fix(b_vals)

def _interp_to_4096(gray12_knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    x_big = np.arange(FULL_POINTS, dtype=float)
    return np.interp(x_big, gray12_knots.astype(float), values.astype(float))

# =======================
# 메인
# =======================
def main(show_plot=True):
    # 1) Low CSV
    df_low = pd.read_csv(INPUT_LOW_CSV)
    _validate_low_csv(df_low)
    R_low = df_low["R_Low"].to_numpy(float)
    G_low = df_low["G_Low"].to_numpy(float)
    B_low = df_low["B_Low"].to_numpy(float)
    if ENFORCE_MONOTONE:
        R_low = _enforce_monotone(R_low)
        G_low = _enforce_monotone(G_low)
        B_low = _enforce_monotone(B_low)

    # 2) High KNOT CSV
    df_k = pd.read_csv(HIGH_KNOT_CSV)
    required_cols = ["Gray8", "Gray12", "R_High", "G_High", "B_High"]
    for c in required_cols:
        if c not in df_k.columns:
            raise ValueError(f"HIGH_KNOT_CSV에 '{c}' 열이 필요합니다.")

    df_k = df_k.sort_values(["Gray12", "Gray8"]).reset_index(drop=True)
    gray8  = df_k["Gray8"].to_numpy(int)
    gray12 = df_k["Gray12"].to_numpy(int)
    Rv     = df_k["R_High"].to_numpy(float)
    Gv     = df_k["G_High"].to_numpy(float)
    Bv     = df_k["B_High"].to_numpy(float)
    _validate_knots(gray12, gray8)
    Rv, Gv, Bv = _apply_edge_value_rules(Rv, Gv, Bv)

    # 3) 보간
    R_high = _interp_to_4096(gray12, Rv)
    G_high = _interp_to_4096(gray12, Gv)
    B_high = _interp_to_4096(gray12, Bv)
    if ENFORCE_MONOTONE:
        R_high = _enforce_monotone(R_high)
        G_high = _enforce_monotone(G_high)
        B_high = _enforce_monotone(B_high)

    # High >= Low + EPS
    R_high = np.maximum(R_high, R_low + EPS_HIGH_OVER_LOW)
    G_high = np.maximum(G_high, G_low + EPS_HIGH_OVER_LOW)
    B_high = np.maximum(B_high, B_low + EPS_HIGH_OVER_LOW)

    # 4) 저장
    out_df = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":  _clip_round_12bit(R_low),
        "R_High": _clip_round_12bit(R_high),
        "G_Low":  _clip_round_12bit(G_low),
        "G_High": _clip_round_12bit(G_high),
        "B_Low":  _clip_round_12bit(B_low),
        "B_High": _clip_round_12bit(B_high),
    })
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Saved: {os.path.abspath(OUTPUT_PATH)}")

    # 5) 시각화 (plt)
    if show_plot:
        plt.figure(figsize=(10,6))
        plt.plot(np.arange(FULL_POINTS), R_high, color='red',  label='R_High (interp)')
        plt.plot(np.arange(FULL_POINTS), G_high, color='green',label='G_High (interp)')
        plt.plot(np.arange(FULL_POINTS), B_high, color='blue', label='B_High (interp)')

        # Knot 포인트 표시
        plt.scatter(gray12, Rv, color='red', marker='o', edgecolors='k', s=30, label='R_High knots')
        plt.scatter(gray12, Gv, color='green', marker='o', edgecolors='k', s=30, label='G_High knots')
        plt.scatter(gray12, Bv, color='blue', marker='o', edgecolors='k', s=30, label='B_High knots')

        plt.title("High LUT (34 knots → 4096 interp)")
        plt.xlabel("Gray Level (12bit)")
        plt.ylabel("Signal Value (12bit)")
        plt.xlim(0, FULL_POINTS-1)
        plt.ylim(0, FULL_POINTS-1)
        plt.legend(fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main(show_plot=True)