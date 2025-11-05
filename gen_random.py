# gen_random_ref_offset.py

import os
import numpy as np
import pandas as pd
from itertools import product


# =========================
# 경로/설정
# =========================
LOW_LUT_CSV   = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_low_values_SIN1300.csv"
HIGH_KNOT_CSV   = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_2_high_34knots_values.csv"
OUTPUT_DIR      = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\CSV"
BASE_NAME       = "LUT_2"

FULL_POINTS = 4096
EPS_HIGH_OVER_LOW = 1
ENFORCE_MONOTONE = True

# ───────────────────────
# 유틸 함수 (단일 LUT 생성용)
# ───────────────────────
def _clip_round_12bit(a): return np.clip(np.rint(a), 0, 4095).astype(np.uint16)
def _enforce_monotone(a):
    a = np.asarray(a, float)
    for i in range(1, len(a)):
        if a[i] < a[i-1]:
            a[i] = a[i-1]
    return a
def _interp_knots_to_4096(gray12, vals):
    return np.interp(np.arange(FULL_POINTS, dtype=float), gray12, vals)

def _apply_offset_with_locks(v, offset, locked):
    v = np.array(v, float)
    for i in range(v.size):
        if i not in locked:
            v[i] += offset
    return v

def _enforce_low_eps(gray12, high_knots, low_curve, eps):
    low_at = np.interp(gray12, np.arange(FULL_POINTS), low_curve)
    need = np.minimum(low_at + eps, 4095.0)
    return np.maximum(high_knots, need)

def _build_single_lut(R_OFFSET, G_OFFSET, B_OFFSET, base_name):
    # Load low
    df_low = pd.read_csv(LOW_LUT_CSV)
    Rl, Gl, Bl = df_low["R_Low"].to_numpy(float), df_low["G_Low"].to_numpy(float), df_low["B_Low"].to_numpy(float)
    Rl, Gl, Bl = _enforce_monotone(Rl), _enforce_monotone(Gl), _enforce_monotone(Bl)

    # Load high knots
    dfk = pd.read_csv(HIGH_KNOT_CSV)
    gray12 = dfk["Gray12"].to_numpy(float)
    Rk, Gk, Bk = dfk["R_High"].to_numpy(float), dfk["G_High"].to_numpy(float), dfk["B_High"].to_numpy(float)

    # Locks: 0, 32, 33 → 0 / 4095 / 4095
    LOCKED = {0, 32, 33}
    FIXED = {0:0.0, 32:4095.0, 33:4095.0}
    for i,v in FIXED.items():
        Rk[i]=v; Gk[i]=v; Bk[i]=v

    # Offset + 제약
    Rk = _apply_offset_with_locks(Rk, R_OFFSET, LOCKED)
    Gk = _apply_offset_with_locks(Gk, G_OFFSET, LOCKED)
    Bk = _apply_offset_with_locks(Bk, B_OFFSET, LOCKED)

    Rk = _enforce_low_eps(gray12, Rk, Rl, EPS_HIGH_OVER_LOW)
    Gk = _enforce_low_eps(gray12, Gk, Gl, EPS_HIGH_OVER_LOW)
    Bk = _enforce_low_eps(gray12, Bk, Bl, EPS_HIGH_OVER_LOW)

    for i,v in FIXED.items():
        Rk[i]=v; Gk[i]=v; Bk[i]=v

    Rk, Gk, Bk = _enforce_monotone(Rk), _enforce_monotone(Gk), _enforce_monotone(Bk)
    Rh = np.clip(_interp_knots_to_4096(gray12, Rk), 0, 4095)
    Gh = np.clip(_interp_knots_to_4096(gray12, Gk), 0, 4095)
    Bh = np.clip(_interp_knots_to_4096(gray12, Bk), 0, 4095)

    Rh = np.maximum(Rh, Rl + EPS_HIGH_OVER_LOW)
    Gh = np.maximum(Gh, Gl + EPS_HIGH_OVER_LOW)
    Bh = np.maximum(Bh, Bl + EPS_HIGH_OVER_LOW)

    out = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":_clip_round_12bit(Rl),"R_High":_clip_round_12bit(Rh),
        "G_Low":_clip_round_12bit(Gl),"G_High":_clip_round_12bit(Gh),
        "B_Low":_clip_round_12bit(Bl),"B_High":_clip_round_12bit(Bh),
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
    out.to_csv(out_path, index=False)
    print(f"[✅] {out_path}")

# ───────────────────────
# 배치 조합 루프
# ───────────────────────
def main():
    offset_values = list(range(-100, 101, 5))  # -100~100 step 5 → 41개
    # 7가지 조합 (R/G/B 각각 포함여부)
    channel_combos = [
        ("R",), ("G",), ("B",),
        ("R","G"), ("R","B"), ("G","B"),
        ("R","G","B")
    ]
    # base_name = os.path.splitext(os.path.basename(HIGH_KNOT_CSV))[0]
    base_name = BASE_NAME

    total = 0
    for combo in channel_combos:
        for off in offset_values:
            R_off = off if "R" in combo else 0
            G_off = off if "G" in combo else 0
            B_off = off if "B" in combo else 0

            # 파일명 규칙
            name_parts = [base_name]
            if "R" in combo: name_parts.append(f"R{off:+d}")
            if "G" in combo: name_parts.append(f"G{off:+d}")
            if "B" in combo: name_parts.append(f"B{off:+d}")
            fname = "_".join(name_parts)

            _build_single_lut(R_off, G_off, B_off, fname)
            total += 1

    print(f"\n[총 {total}개 LUT 생성 완료 ✅]")

if __name__ == "__main__":
    main()
