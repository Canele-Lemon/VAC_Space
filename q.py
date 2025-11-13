# interpolate_gray8_to_gray12.py

import os
import numpy as np
import pandas as pd
import tempfile

# ----- GUI -----
try:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    use_gui = True
except Exception:
    use_gui = False

# ===== 공통 상수 =====
FULL_POINTS = 4096        # 12bit LUT
EPS_HIGH_OVER_LOW = 1.0   # High ≥ Low + 1 제약


# ===== 유틸 =====
def _clip_round_12bit(a):
    return np.clip(np.rint(a), 0, 4095).astype(np.uint16)

def _enforce_monotone(a):
    a = np.asarray(a, float).copy()
    for i in range(1, len(a)):
        if a[i] < a[i - 1]:
            a[i] = a[i - 1]
    return a

def _enforce_with_locks_12bit(high, low):
    """
    gen_random_ref_offset.py 에서 사용한 방식으로
    12bit 인덱스 기준 lock + 단조 + low+eps 제약을 적용
    """
    high = np.asarray(high, float).copy()
    low  = np.asarray(low, float).copy()

    # 중요 잠금 지점
    LOCK_VALS = {
        0: 0.0,        # Gray8=0,1 이 맵핑되는 12bit=0
        1: 0.0,        # (1도 동일)
        4092: 4092.0,  # Gray8=254
        4095: 4095.0,  # Gray8=255
    }
    lock_idx = np.array(list(LOCK_VALS.keys()))

    # 1) 잠금 값 우선 적용
    for j, v in LOCK_VALS.items():
        high[j] = v

    # 2) 잠금 제외 영역에서 low + eps 적용
    mask = np.ones_like(high, dtype=bool)
    mask[lock_idx] = False
    high[mask] = np.maximum(high[mask], low[mask] + EPS_HIGH_OVER_LOW)

    # 3) 중간구간 단조 증가
    start, end = 2, 4091
    high[start:end+1] = np.maximum.accumulate(high[start:end+1])

    # 4) 상한 clamp
    high[:4092] = np.minimum(high[:4092], high[4092])
    high[4092:4096] = np.maximum.accumulate(high[4092:4096])
    high[4092:4096] = np.minimum(high[4092:4096], high[4095])

    # 5) 잠금 값 다시 덮기 (보호)
    for j, v in LOCK_VALS.items():
        high[j] = v

    return high


# ===== Low 12bit LUT =====
def load_low_gray12(csv_path):
    df = pd.read_csv(csv_path)

    if "Gray12" not in df.columns:
        df.insert(0, "Gray12", np.arange(FULL_POINTS))

    cols = {}
    for c in ["R_Low", "G_Low", "B_Low"]:
        cand = [c, c.lower(), c.replace("_", ""), c + "_full"]
        for x in cand:
            if x in df.columns:
                cols[c] = x
                break

    if len(cols) != 3:
        raise ValueError("Low LUT CSV에서 R_Low/G_Low/B_Low 열을 찾을 수 없습니다.")

    # 정렬 + 단조 보정
    df = df.sort_values("Gray12")
    Rl = _enforce_monotone(df[cols["R_Low"]].to_numpy(float))
    Gl = _enforce_monotone(df[cols["G_Low"]].to_numpy(float))
    Bl = _enforce_monotone(df[cols["B_Low"]].to_numpy(float))

    return pd.DataFrame({
        "Gray12": np.arange(FULL_POINTS),
        "R_Low_full": Rl,
        "G_Low_full": Gl,
        "B_Low_full": Bl,
    })


# ===== High (Sparse) → 4096 보간 =====
def load_sparse_high(csv_path):
    df = pd.read_csv(csv_path)

    required = ["Gray8", "Gray12", "R_High", "G_High", "B_High"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"필수 컬럼 없음: {miss}")

    df["Gray8"]  = pd.to_numeric(df["Gray8"], errors="coerce").astype("Int64")
    df["Gray12"] = pd.to_numeric(df["Gray12"], errors="coerce").astype("Int64")
    for c in ["R_High", "G_High", "B_High"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Gray8 = 254 → Gray12 = 4092
    df.loc[df["Gray8"] == 254, "Gray12"] = 4092

    # 같은 Gray12 중복 시 "마지막 행" 유지
    df = df.sort_values(["Gray12", "Gray8"])
    df = df.groupby("Gray12", as_index=False).tail(1)
    df = df.sort_values("Gray12")

    return df


def interp_sparse_to_full_4096(df_sparse):
    x = df_sparse["Gray12"].to_numpy(float)

    full_j = np.arange(FULL_POINTS)

    R = np.interp(full_j, x, df_sparse["R_High"].to_numpy(float))
    G = np.interp(full_j, x, df_sparse["G_High"].to_numpy(float))
    B = np.interp(full_j, x, df_sparse["B_High"].to_numpy(float))

    return pd.DataFrame({
        "Gray12": full_j,
        "R_High_full": R,
        "G_High_full": G,
        "B_High_full": B,
    })


# ===== 최종 생성 =====
def build_full_4096_table(sparse_csv, low_csv):

    df_low  = load_low_gray12(low_csv)
    df_high = load_sparse_high(sparse_csv)
    df_h12  = interp_sparse_to_full_4096(df_high)

    # numpy로 꺼내기
    Rl = df_low["R_Low_full"].to_numpy(float)
    Gl = df_low["G_Low_full"].to_numpy(float)
    Bl = df_low["B_Low_full"].to_numpy(float)

    Rh = df_h12["R_High_full"].to_numpy(float)
    Gh = df_h12["G_High_full"].to_numpy(float)
    Bh = df_h12["B_High_full"].to_numpy(float)

    # ★ gen_random_ref_offset 방식의 12bit 제약 적용 ★
    Rh = _enforce_with_locks_12bit(Rh, Rl)
    Gh = _enforce_with_locks_12bit(Gh, Gl)
    Bh = _enforce_with_locks_12bit(Bh, Bl)

    return pd.DataFrame({
        "Gray12": np.arange(FULL_POINTS),
        "R_Low": _clip_round_12bit(Rl),
        "R_High": _clip_round_12bit(Rh),
        "G_Low": _clip_round_12bit(Gl),
        "G_High": _clip_round_12bit(Gh),
        "B_Low": _clip_round_12bit(Bl),
        "B_High": _clip_round_12bit(Bh),
    })


# ===== Main =====
def main():
    if use_gui:
        root = tk.Tk(); root.withdraw()

    if use_gui:
        sparse_csv = askopenfilename(title="Sparse High CSV 선택")
    else:
        sparse_csv = input("Sparse High CSV 경로: ").strip()
    if not sparse_csv:
        print("@INFO: 입력 안함. 종료")
        return

    # FULL 4096 테이블 생성
    df = build_full_4096_table(sparse_csv, LOW_LUT_CSV)

    # 임시파일 저장
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix="_LUT_Gray12.csv") as tmp:
        tmp_path = tmp.name

    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 4096 LUT 생성: {tmp_path}")

    # 자동 실행
    try:
        os.startfile(tmp_path)
    except:
        pass


if __name__ == "__main__":
    main()