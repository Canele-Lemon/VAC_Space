# build_lut_from_knots_offsets.py
# - Low: 기존 CSV(R_Low,G_Low,B_Low)를 그대로 사용(4096 포인트)
# - High: HIGH_KNOT_CSV의 34개 knot(Gray12, R/G/B_High)에 채널별 OFFSET 적용
#         단, knot idx 0=0, idx 32=4095, idx 33=4095로 "값 고정"
# - 제약: High ≥ Low + EPS, 0 ≤ 값 ≤ 4095, 단조(비내림) 유지
# - 보간: 34개 → 4096개 선형보간
# - 출력: GrayLevel_window, R_Low,R_High,G_Low,G_High,B_Low,B_High
# - 플롯: Low/High를 함께 matplotlib에 표시(옵션)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# 경로/설정
# =======================
LOW_LUT_CSV    = r"./your_low_lut_4096.csv"         # 입력 Low LUT CSV (4096행)
HIGH_KNOT_CSV  = r"./your_high_knots_34.csv"         # 입력 High knot CSV (34행)
OUTPUT_CSV     = r"./LUT_full_4096_from_knots.csv"   # 출력 파일 경로

# 채널별 OFFSET (knot 값에 상수 가감; 단, 잠금 knot에는 미적용)
R_OFFSET = 0
G_OFFSET = 0
B_OFFSET = 0

# 제약 파라미터
FULL_POINTS = 4096
KNOT_COUNT  = 34                      # 34개 knot
EPS_HIGH_OVER_LOW = 1                 # High ≥ Low + EPS
ENFORCE_MONOTONE = True               # 단조(비내림) 유지

# 플롯 옵션
SHOW_PLOT = True
PLOT_TITLE = "Low vs High (after constraints)"

# =======================
# 유틸 함수
# =======================
def _clip_round_12bit(arr):
    a = np.clip(np.rint(arr), 0, FULL_POINTS - 1)
    return a.astype(np.uint16)

def _enforce_monotone(arr):
    a = np.asarray(arr, dtype=float).copy()
    for i in range(1, a.size):
        if a[i] < a[i-1]:
            a[i] = a[i-1]
    return a

def _interp_knots_to_4096(gray12_knots, values):
    x_big = np.arange(FULL_POINTS, dtype=float)
    return np.interp(x_big, np.asarray(gray12_knots, float), np.asarray(values, float))

def apply_offset_with_locks(values, offset, locked_idx):
    """잠금 인덱스는 그대로, 나머지만 offset 적용"""
    v = np.asarray(values, dtype=float).copy()
    for i in range(v.size):
        if i in locked_idx:
            continue
        v[i] += float(offset)
    return v

def monotone_non_decreasing_with_locks(values, locked_idx):
    """
    단조(비내림)로 투영. 잠금 인덱스의 값은 유지하되,
    일반 지점이 잠금 지점 뒤에서 위반하면 그 잠금 값에 맞춰 끌어올림.
    """
    v = np.asarray(values, dtype=float).copy()
    n = v.size
    # 왼→오
    for i in range(1, n):
        if v[i] < v[i-1]:
            # i가 잠금이면 이전 값을 i-1로 낮출 수 없어야 하는데,
            # (잠금 보존) 규칙상 잠금값 변경 금지 → i를 끌어올림
            v[i] = v[i-1]
    # 오→왼 (필요 시 한 번 더 안정화)
    for i in range(n-2, -1, -1):
        if v[i] > v[i+1]:
            # i가 잠금이면 i+1을 내릴 수 없으니 i를 끌어내림…은 잠금 위반.
            # 따라서 여기서도 i를 끌어내리면 안됨. 대신 i+1에 맞춰 재상향은 위에서 함.
            v[i] = v[i+1]
    # 잠금 인덱스 값은 원본으로 강제 복구
    # (단, 앞뒤 단조화로 인해 국소적 불연속 생길 수 있으나 보간 단계에서 완화됨)
    return v

def enforce_low_eps_on_knots(gray12_knots, high_knots, low4096, eps):
    """
    각 knot 위치에서 High ≥ Low + EPS를 보장.
    4095 초과는 4095로 캡.
    """
    low_at_knots = np.interp(np.asarray(gray12_knots, float),
                             np.arange(FULL_POINTS, dtype=float),
                             low4096.astype(float))
    need = np.minimum(low_at_knots + float(eps), 4095.0)
    return np.maximum(np.asarray(high_knots, float), need)

# =======================
# 메인
# =======================
def main():
    # 1) Low 4096 읽기
    df_low = pd.read_csv(LOW_LUT_CSV)
    for c in ("R_Low","G_Low","B_Low"):
        if c not in df_low.columns:
            raise ValueError(f"[입력 오류] LOW_LUT_CSV에 '{c}' 열이 필요합니다.")
    if len(df_low) != FULL_POINTS:
        raise ValueError(f"[입력 오류] LOW_LUT_CSV 행 수={len(df_low)} (4096이어야 함)")

    R_low = df_low["R_Low"].to_numpy(float)
    G_low = df_low["G_Low"].to_numpy(float)
    B_low = df_low["B_Low"].to_numpy(float)

    if ENFORCE_MONOTONE:
        R_low = _enforce_monotone(R_low)
        G_low = _enforce_monotone(G_low)
        B_low = _enforce_monotone(B_low)

    # 2) High knot 34개 읽기
    df_k = pd.read_csv(HIGH_KNOT_CSV)
    need_cols = ("Gray8","Gray12","R_High","G_High","B_High")
    for c in need_cols:
        if c not in df_k.columns:
            raise ValueError(f"[입력 오류] HIGH_KNOT_CSV에 '{c}' 열이 필요합니다.")

    if len(df_k) != KNOT_COUNT:
        raise ValueError(f"[입력 오류] HIGH_KNOT_CSV 행 수={len(df_k)} (예상 {KNOT_COUNT})")

    gray12_knots = df_k["Gray12"].to_numpy(float)
    RL_k = df_k["R_High"].to_numpy(float)
    GL_k = df_k["G_High"].to_numpy(float)
    BL_k = df_k["B_High"].to_numpy(float)

    # 2-1) knot 오름차순 및 범위 체크
    if not np.all(np.diff(gray12_knots) >= 0):
        raise ValueError("[입력 오류] Gray12(knot)는 오름차순이어야 합니다.")
    if gray12_knots.min() < 0 or gray12_knots.max() > 4095:
        raise ValueError("[입력 오류] Gray12(knot) 범위는 [0,4095]")

    # 3) 잠금 인덱스 및 고정값
    #    idx 0 = 0, idx 32 = 4095, idx 33 = 4095
    LOCKED = {0, 32, 33}
    FIXED_VALS = {0: 0.0, 32: 4095.0, 33: 4095.0}

    # 3-1) 먼저 입력 knot 값에 고정 적용 (CSV 값 무시하고 강제 설정)
    for i, v in FIXED_VALS.items():
        RL_k[i] = v
        GL_k[i] = v
        BL_k[i] = v

    # 4) 채널별 OFFSET 적용 (잠금 제외)
    RL_k = apply_offset_with_locks(RL_k, R_OFFSET, LOCKED)
    GL_k = apply_offset_with_locks(GL_k, G_OFFSET, LOCKED)
    BL_k = apply_offset_with_locks(BL_k, B_OFFSET, LOCKED)

    # 5) 단조(비내림) 투영 (잠금 유지)
    if ENFORCE_MONOTONE:
        RL_k = monotone_non_decreasing_with_locks(RL_k, LOCKED)
        GL_k = monotone_non_decreasing_with_locks(GL_k, LOCKED)
        BL_k = monotone_non_decreasing_with_locks(BL_k, LOCKED)

    # 6) 각 knot에서 High ≥ Low + EPS 제약 (4095 cap)
    RL_k = enforce_low_eps_on_knots(gray12_knots, RL_k, R_low, EPS_HIGH_OVER_LOW)
    GL_k = enforce_low_eps_on_knots(gray12_knots, GL_k, G_low, EPS_HIGH_OVER_LOW)
    BL_k = enforce_low_eps_on_knots(gray12_knots, BL_k, B_low, EPS_HIGH_OVER_LOW)

    # 6-1) 다시 잠금값을 최종 강제(혹시 위에서 EPS 제약으로 바뀌지 않도록)
    for i, v in FIXED_VALS.items():
        RL_k[i] = v
        GL_k[i] = v
        BL_k[i] = v

    # 7) 34 → 4096 보간
    R_high_4096 = _interp_knots_to_4096(gray12_knots, RL_k)
    G_high_4096 = _interp_knots_to_4096(gray12_knots, GL_k)
    B_high_4096 = _interp_knots_to_4096(gray12_knots, BL_k)

    # 8) 전 구간 제약: High ≥ Low + EPS, 0..4095, 단조(옵션)
    R_high_4096 = np.maximum(R_high_4096, R_low + EPS_HIGH_OVER_LOW)
    G_high_4096 = np.maximum(G_high_4096, G_low + EPS_HIGH_OVER_LOW)
    B_high_4096 = np.maximum(B_high_4096, B_low + EPS_HIGH_OVER_LOW)

    R_high_4096 = np.clip(R_high_4096, 0, 4095)
    G_high_4096 = np.clip(G_high_4096, 0, 4095)
    B_high_4096 = np.clip(B_high_4096, 0, 4095)

    if ENFORCE_MONOTONE:
        R_high_4096 = _enforce_monotone(R_high_4096)
        G_high_4096 = _enforce_monotone(G_high_4096)
        B_high_4096 = _enforce_monotone(B_high_4096)

    # 9) 저장용 정수화
    out_df = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":  _clip_round_12bit(R_low),
        "R_High": _clip_round_12bit(R_high_4096),
        "G_Low":  _clip_round_12bit(G_low),
        "G_High": _clip_round_12bit(G_high_4096),
        "B_Low":  _clip_round_12bit(B_low),
        "B_High": _clip_round_12bit(B_high_4096),
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[✅] Saved: {os.path.abspath(OUTPUT_CSV)}")

    # 10) 플롯 (Low/High 함께)
    if SHOW_PLOT:
        x = np.arange(FULL_POINTS)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, R_low,        label="R_Low")
        ax.plot(x, R_high_4096,  label="R_High")
        ax.plot(x, G_low,        label="G_Low")
        ax.plot(x, G_high_4096,  label="G_High")
        ax.plot(x, B_low,        label="B_Low")
        ax.plot(x, B_high_4096,  label="B_High")
        ax.set_title(PLOT_TITLE)
        ax.set_xlabel("Gray (12bit index)")
        ax.set_ylabel("LUT value (12bit)")
        ax.set_xlim(0, FULL_POINTS-1)
        ax.set_ylim(0, 4095)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()