# build_287_luts_named.py
# LUT_1_R+10_G+10_B+10.csv 방식으로 저장

import os
import numpy as np
import pandas as pd

# =========================
# 경로/설정
# =========================
INPUT_LOW_CSV   = r"./your_low_lut_4096.csv"        # 기존 Low 채널 CSV
HIGH_KNOT_CSV   = r"./high_knots_34pts.csv"         # (8bit_gray, 12bit_gray, R_High, G_High, B_High)
OUTPUT_DIR      = r"./LUT_SWEEP_287"                # 출력 폴더
BASE_NAME       = "LUT_1"                           # 기준 LUT 이름 (고정)

FULL_POINTS = 4096
EPS_HIGH_OVER_LOW = 1
ENFORCE_MONOTONE = True

OFFSETS = list(range(-100, 101, 5))  # -100~100 step 5
CHANNEL_COMBOS = {
    "R":   ("R_High",),
    "G":   ("G_High",),
    "B":   ("B_High",),
    "RG":  ("R_High", "G_High"),
    "RB":  ("R_High", "B_High"),
    "GB":  ("G_High", "B_High"),
    "RGB": ("R_High", "G_High", "B_High"),
}

# =========================
# 유틸
# =========================
def _enforce_monotone(a):
    a = np.asarray(a, float).copy()
    for i in range(1, a.size):
        if a[i] < a[i - 1]:
            a[i] = a[i - 1]
    return a

def _clip_round_12bit(a):
    return np.clip(np.rint(a), 0, 4095).astype(np.uint16)

def _interp_to_4096(x_small, y_small):
    x_big = np.arange(FULL_POINTS, dtype=float)
    return np.interp(x_big, x_small.astype(float), y_small.astype(float))

def _make_filename(base_name, channels, offset):
    """예: LUT_1_R+10_G+10.csv"""
    parts = [base_name]
    for ch in ("R_High", "G_High", "B_High"):
        if ch in channels:
            c = ch[0]  # R/G/B
            parts.append(f"{c}{offset:+d}")
    return "_".join(parts) + ".csv"

# =========================
# 데이터 로드
# =========================
def load_low_curves(path):
    df = pd.read_csv(path)
    return (
        df["R_Low"].to_numpy(float),
        df["G_Low"].to_numpy(float),
        df["B_Low"].to_numpy(float),
    )

def load_high_knots(path):
    df = pd.read_csv(path)
    gray12 = df.iloc[:, 1].to_numpy(float)
    RH = df.iloc[:, 2].to_numpy(float)
    GH = df.iloc[:, 3].to_numpy(float)
    BH = df.iloc[:, 4].to_numpy(float)
    return gray12, {"R_High": RH, "G_High": GH, "B_High": BH}

# =========================
# 메인 로직
# =========================
def build_and_save_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    R_low, G_low, B_low = load_low_curves(INPUT_LOW_CSV)
    gray12, base_high_knots = load_high_knots(HIGH_KNOT_CSV)

    total = 0
    for combo_name, channels in CHANNEL_COMBOS.items():
        for offset in OFFSETS:
            # 각 채널별 knot 준비
            knots_vals = {}
            for ch in ("R_High", "G_High", "B_High"):
                vals = base_high_knots[ch].copy()
                if ch in channels:
                    vals = np.clip(vals + offset, 0, 4095)
                knots_vals[ch] = vals

            # 4096포인트 보간
            R_high = _interp_to_4096(gray12, knots_vals["R_High"])
            G_high = _interp_to_4096(gray12, knots_vals["G_High"])
            B_high = _interp_to_4096(gray12, knots_vals["B_High"])

            # 단조/제약
            if ENFORCE_MONOTONE:
                R_high = _enforce_monotone(R_high)
                G_high = _enforce_monotone(G_high)
                B_high = _enforce_monotone(B_high)

            R_high = np.maximum(R_high, R_low + EPS_HIGH_OVER_LOW)
            G_high = np.maximum(G_high, G_low + EPS_HIGH_OVER_LOW)
            B_high = np.maximum(B_high, B_low + EPS_HIGH_OVER_LOW)

            R_high = _clip_round_12bit(R_high)
            G_high = _clip_round_12bit(G_high)
            B_high = _clip_round_12bit(B_high)

            out_df = pd.DataFrame({
                "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
                "R_Low":  _clip_round_12bit(R_low),
                "R_High": R_high,
                "G_Low":  _clip_round_12bit(G_low),
                "G_High": G_high,
                "B_Low":  _clip_round_12bit(B_low),
                "B_High": B_high,
            })

            # 파일명 생성
            fname = _make_filename(BASE_NAME, channels, offset)
            out_path = os.path.join(OUTPUT_DIR, fname)
            out_df.to_csv(out_path, index=False)
            total += 1

    print(f"[✅] LUT 생성 완료 — 총 {total}개 파일 생성됨 (예상 287)")
    print(f"[📂] 경로: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    build_and_save_all()