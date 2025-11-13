# gen_random_ref_offset.py

import os
import tempfile
import numpy as np
import pandas as pd
from itertools import product


# =========================
# 경로/설정
# =========================
LOW_LUT_CSV   = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_low_values_SIN1300_254gray를4092로변경.csv"
HIGH_KNOT_CSV   = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_3_high_256knots_values.csv"
OUTPUT_DIR      = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\CSV_LUT_3"
BASE_NAME       = "LUT_3"

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

def _build_single_lut(R_OFFSET, G_OFFSET, B_OFFSET, base_name=None):
    # Load low
    df_low = pd.read_csv(LOW_LUT_CSV)
    Rl, Gl, Bl = df_low["R_Low"].to_numpy(float), df_low["G_Low"].to_numpy(float), df_low["B_Low"].to_numpy(float)
    Rl, Gl, Bl = _enforce_monotone(Rl), _enforce_monotone(Gl), _enforce_monotone(Bl)

    # Load high knots
    dfk = pd.read_csv(HIGH_KNOT_CSV)
    gray8 = dfk["Gray8"].to_numpy(int)
    gray12 = dfk["Gray12"].to_numpy(float)
    Rk, Gk, Bk = dfk["R_High"].to_numpy(float), dfk["G_High"].to_numpy(float), dfk["B_High"].to_numpy(float)

    # Locks: Gray 0, 1  → 0 / Gray 254 → 4092 / Gray 255  → 4095
    mask_gray_0 = (gray8 == 0)
    mask_gray_1 = (gray8 == 1)
    mask_gray254 = (gray8 == 254)
    mask_gray255 = (gray8 == 255)
    
    lock_mask = mask_gray_0 | mask_gray_1 | mask_gray254 | mask_gray255
    lock_indices = np.where(lock_mask)[0]
    
    FIXED = {}
    for idx in lock_indices:
        g = gray8[idx]
        if g in (0, 1):
            val == 0.0
        elif g == 254:
            val == 4092.0 
        elif g == 255:
            val == 4095.0
        else:
            continue
        
    for idx, val in FIXED.items():
        Rk[idx] = val
        Gk[idx] = val 
        Bk[idx] = val
    
    LOCKED = set(FIXED.keys())

    # Offset + 제약
    Rk = _apply_offset_with_locks(Rk, R_OFFSET, LOCKED)
    Gk = _apply_offset_with_locks(Gk, G_OFFSET, LOCKED)
    Bk = _apply_offset_with_locks(Bk, B_OFFSET, LOCKED)
    for idx, val in FIXED.items():
        Rk[idx] = val
        Gk[idx] = val 
        Bk[idx] = val

    Rk = _enforce_low_eps(gray12, Rk, Rl, EPS_HIGH_OVER_LOW)
    Gk = _enforce_low_eps(gray12, Gk, Gl, EPS_HIGH_OVER_LOW)
    Bk = _enforce_low_eps(gray12, Bk, Bl, EPS_HIGH_OVER_LOW)
    for idx, val in FIXED.items():
        Rk[idx] = val
        Gk[idx] = val 
        Bk[idx] = val

    Rk = _enforce_monotone(Rk) 
    Gk = _enforce_monotone(Gk)
    Bk = _enforce_monotone(Bk)
    for idx, val in FIXED.items():
        Rk[idx] = val
        Gk[idx] = val 
        Bk[idx] = val
        
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
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # out_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
    # out.to_csv(out_path, index=False)
    
    return out

# ───────────────────────
# 배치 조합 루프
# ───────────────────────
def main():
    # 테스트: offset = +500 한 세트만 생성
    R_off = 500
    G_off = 500
    B_off = 500

    print(f"[TEST] Generating single LUT: R={R_off}, G={G_off}, B={B_off}")

    df = _build_single_lut(R_off, G_off, B_off)   # ← DataFrame 생성

    # 임시파일 생성
    tmp = tempfile.NamedTemporaryFile(delete=False,
                                      suffix=f"_LUT_R{R_off}_G{G_off}_B{B_off}.csv")
    tmp_path = tmp.name
    tmp.close()

    # CSV로 저장
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 임시 CSV 생성: {tmp_path}")

    # Windows에서 자동 열기
    try:
        os.startfile(tmp_path)
    except Exception:
        pass

    print("[DONE] Single LUT test completed.")    
    
    
    # # 1) offset 범위 (원하시면 여기서 -900~900, step 50으로 바꾸면 됨)
    # offset_values = list(range(-500, 501, 25))  # 예: -100 ~ 100, step 10

    # channel_combos = [
    #     ("R",), ("G",), ("B",),
    #     ("R","G"), ("R","B"), ("G","B"),
    #     ("R","G","B")
    # ]
    # base_name = BASE_NAME

    # # 2) 먼저 (R,G,B) offset 조합을 unique하게 모으기
    # unique_offsets = set()

    # for combo in channel_combos:
    #     for off in offset_values:
    #         R_off = off if "R" in combo else 0
    #         G_off = off if "G" in combo else 0
    #         B_off = off if "B" in combo else 0
    #         unique_offsets.add((R_off, G_off, B_off))

    # # 혹시 offset_values에 0이 없어도 대비해서 기준 (0,0,0)은 무조건 추가
    # unique_offsets.add((0, 0, 0))

    # print(f"[INFO] unique LUT set: {len(unique_offsets)} combinations")

    # total = 0
    # for (R_off, G_off, B_off) in sorted(unique_offsets):
    #     name_parts = [base_name]

    #     # 기준 LUT라면 Base라는 suffix만 붙이기
    #     if R_off == 0 and G_off == 0 and B_off == 0:
    #         name_parts.append("Base")
    #     else:
    #         if R_off != 0:
    #             name_parts.append(f"R{R_off:+d}")
    #         if G_off != 0:
    #             name_parts.append(f"G{G_off:+d}")
    #         if B_off != 0:
    #             name_parts.append(f"B{B_off:+d}")

    #     fname = "_".join(name_parts)
    #     _build_single_lut(R_off, G_off, B_off, fname)
    #     total += 1

    # print(f"\n[총 {total}개 LUT 생성 완료 ✅]")

if __name__ == "__main__":
    main()

아래 에러가떠요
[TEST] Generating single LUT: R=500, G=500, B=500
Traceback (most recent call last):
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\x_generation\gen_random_ref_offset.py", line 212, in <module>
    main()
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\x_generation\gen_random_ref_offset.py", line 143, in main
    df = _build_single_lut(R_off, G_off, B_off)   # ← DataFrame 생성
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\x_generation\gen_random_ref_offset.py", line 72, in _build_single_lut
    val == 0.0
UnboundLocalError: local variable 'val' referenced before assignment
