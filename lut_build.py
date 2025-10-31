# lut_build_high_from_33_keep_low_gray.py
# - Low: 기존 CSV에서 그대로 사용 (4096포인트)
# - High: 공통 KNOTS(길이 자유) → 선형 보간
# - 첫 열: GrayLevel_window (0~4095)
# - 출력 열: GrayLevel_window, R_Low,R_High,G_Low,G_High,B_Low,B_High

import numpy as np
import pandas as pd
import os

# =======================
# 경로/설정
# =======================
INPUT_LOW_CSV  = r"./your_low_lut_4096.csv"         # 기존 Low 채널 CSV
OUTPUT_PATH    = r"./LUT_full_4096_high_from_knots_with_gray.csv"

FULL_POINTS = 4096
ENFORCE_MONOTONE = True
EPS_HIGH_OVER_LOW = 1

LOW_COLS = ["R_Low", "G_Low", "B_Low"]              # 입력 CSV에 필요한 열

# ---- High 33포인트(또는 그 이상) 값 로드 관련 ----
CHANNEL_SOURCE = "csv"      # "csv" 또는 "inline"
HIGH_KNOT_CSV  = r"./high_knots_values.csv"  # CHANNEL_SOURCE="csv"일 때 사용
CSV_HAS_GRAYINDEX = True    # CSV에 GrayIndex 열이 있으면 True

# =======================
# 1) KNOTS 지정 (원하는 리스트로 수정)
#    ※ KNOT_COUNT는 len(KNOTS)로 자동 계산됩니다.
# =======================
KNOTS = [
    0,    112,  240,  368,  499,  626,  758,  889,  1020, 1148, 1279,
    1410, 1522, 1653, 1780, 1911, 2043, 2175, 2304, 2437, 2570, 2699,
    2812, 2945, 3078, 3211, 3340, 3473, 3606, 3739, 3868, 4000, 4092, 4095
]
KNOT_COUNT = len(KNOTS)

# 1번 knot=0, 33·34번 knot=4095 규칙을 인덱스로 강제 적용하기 위한 설정
FORCE_FIRST_ZERO = True
FORCE_4095_IDX = [32, 33]   # 33번, 34번 knot → 0-based index 32, 33

# =======================
# 2) High 채널용 값 (inline 사용 시)
# =======================
def _auto_baseline_values(offset=1):
    x_small = np.array(KNOTS, dtype=float)
    y_small = x_small + offset
    return np.clip(y_small, 0, FULL_POINTS - 1).tolist()

CHANNELS_INLINE = {
    "R_High": _auto_baseline_values(offset=1),
    "G_High": _auto_baseline_values(offset=1),
    "B_High": _auto_baseline_values(offset=1),
}

# =======================
# 유틸
# =======================
def _interp_knots_to_4096(knots, values):
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

def _validate_knots(knots):
    if list(sorted(knots)) != list(knots):
        raise ValueError("KNOTS는 오름차순이어야 합니다.")
    if knots[0] < 0 or knots[-1] > FULL_POINTS - 1:
        raise ValueError("KNOTS 값은 0~4095 범위여야 합니다.")

def _apply_edge_constraints(values):
    """엣지 고정 규칙: 1번 knot=0, 33·34번 knot=4095"""
    v = list(values)
    # 1) 첫 knot가 gray=0일 때 값=0 강제
    if FORCE_FIRST_ZERO and KNOTS[0] == 0:
        v[0] = 0
    # 2) 33, 34번 knot 강제 4095 (인덱스 존재할 때만)
    for idx in FORCE_4095_IDX:
        if 0 <= idx < KNOT_COUNT:
            v[idx] = 4095
    return v

def _load_channels_from_csv(csv_path, knots, has_gray=True):
    """
    CSV 포맷:
    - (권장) GrayIndex, R_High, G_High, B_High
      → GrayIndex가 KNOTS와 동일한지 검증
    - (간단) R_High, G_High, B_High (행 수는 len(KNOTS))
    """
    df = pd.read_csv(csv_path)
    need_cols = ["R_High", "G_High", "B_High"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV에 '{c}' 열이 필요합니다.")

    if has_gray:
        if "GrayIndex" not in df.columns:
            raise ValueError("CSV_HAS_GRAYINDEX=True 이면 'GrayIndex' 열이 필요합니다.")
        df = df.sort_values("GrayIndex").reset_index(drop=True)
        gray_csv = df["GrayIndex"].astype(int).tolist()
        if len(gray_csv) != len(knots):
            raise ValueError(f"CSV 행수({len(gray_csv)}) != len(KNOTS)({len(knots)})")
        if list(gray_csv) != list(knots):
            raise ValueError("CSV의 GrayIndex가 KNOTS와 일치하지 않습니다.")
        Rv = df["R_High"].astype(float).tolist()
        Gv = df["G_High"].astype(float).tolist()
        Bv = df["B_High"].astype(float).tolist()
    else:
        # 행 수가 len(KNOTS)와 같아야 하며, 순서는 KNOTS 순서를 따른다고 가정
        if len(df) != len(knots):
            raise ValueError(f"CSV 행수({len(df)}) != len(KNOTS)({len(knots)})")
        Rv = df["R_High"].astype(float).tolist()
        Gv = df["G_High"].astype(float).tolist()
        Bv = df["B_High"].astype(float).tolist()

    # 엣지 규칙 적용
    Rv = _apply_edge_constraints(Rv)
    Gv = _apply_edge_constraints(Gv)
    Bv = _apply_edge_constraints(Bv)

    # 범위 클립
    Rv = np.clip(Rv, 0, FULL_POINTS - 1).tolist()
    Gv = np.clip(Gv, 0, FULL_POINTS - 1).tolist()
    Bv = np.clip(Bv, 0, FULL_POINTS - 1).tolist()

    return {"R_High": Rv, "G_High": Gv, "B_High": Bv}

def _get_high_channels():
    if CHANNEL_SOURCE.lower() == "csv":
        return _load_channels_from_csv(HIGH_KNOT_CSV, KNOTS, has_gray=CSV_HAS_GRAYINDEX)
    # inline: 엣지 규칙을 여기서도 강제
    out = {}
    for k in ("R_High", "G_High", "B_High"):
        vals = CHANNELS_INLINE[k]
        if len(vals) != KNOT_COUNT:
            raise ValueError(f"{k} 길이({len(vals)}) != len(KNOTS)({KNOT_COUNT})")
        out[k] = _apply_edge_constraints(vals)
    return out

# =======================
# 메인
# =======================
def main():
    _validate_knots(KNOTS)

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

    # 2) High 33포인트(또는 지정 길이) 값 준비 (CSV 또는 inline)
    CHANNELS = _get_high_channels()

    # 3) 보간하여 4096 포인트 생성
    R_high = _interp_knots_to_4096(KNOTS, CHANNELS["R_High"])
    G_high = _interp_knots_to_4096(KNOTS, CHANNELS["G_High"])
    B_high = _interp_knots_to_4096(KNOTS, CHANNELS["B_High"])

    if ENFORCE_MONOTONE:
        R_high = _enforce_monotone(R_high)
        G_high = _enforce_monotone(G_high)
        B_high = _enforce_monotone(B_high)

    # High >= Low + EPS
    R_high = np.maximum(R_high, R_low + EPS_HIGH_OVER_LOW)
    G_high = np.maximum(G_high, G_low + EPS_HIGH_OVER_LOW)
    B_high = np.maximum(B_high, B_low + EPS_HIGH_OVER_LOW)

    # 4) 결과 DataFrame
    out_df = pd.DataFrame({
        "GrayLevel_window": np.arange(FULL_POINTS, dtype=np.uint16),
        "R_Low":  _clip_round_12bit(R_low),
        "R_High": _clip_round_12bit(R_high),
        "G_Low":  _clip_round_12bit(G_low),
        "G_High": _clip_round_12bit(G_high),
        "B_Low":  _clip_round_12bit(B_low),
        "B_High": _clip_round_12bit(B_high),
    })

    # 5) 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Saved: {os.path.abspath(OUTPUT_PATH)}")
    print(f" - #KNOTS : {KNOT_COUNT}")
    print(f" - Output : {len(out_df)} rows (expected {FULL_POINTS})")

if __name__ == "__main__":
    main()