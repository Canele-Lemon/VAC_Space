# interpolate_lut_j_to_4096.py

import os
import sys
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
# ---------------

# ===== 공통 상수/유틸 =====
FULL_POINTS = 4096
EPS_HIGH_OVER_LOW = 1.0  # High ≥ Low + 1 보장

def _clip_round_12bit(a):
    return np.clip(np.rint(a), 0, 4095).astype(np.uint16)

def _enforce_monotone(a):
    """
    1D 배열 a에 대해 앞에서 뒤로 갈수록 감소하지 않도록 (비감소) 강제.
    """
    a = np.asarray(a, float).copy()
    for i in range(1, len(a)):
        if a[i] < a[i-1]:
            a[i] = a[i-1]
    return a

def _enforce_with_locks_12bit(high, low, eps=1.0):
    """
    gen_random_ref_offset.py에서 사용한 로직과 동일한 컨셉.

    high, low: 길이 4096 배열
    잠금 인덱스: 0, 1, 4092, 4095
      - high[0] = high[1] = 0
      - high[4092] = 4092
      - high[4095] = 4095
    그 외 인덱스:
      - high[j] >= low[j] + eps
      - 전체 단조 증가(비감소)
      - 단, 잠금 인덱스는 그대로 두고,
        잠금 인덱스 주변 값이 잠금값을 넘으면 잠금값으로 끌어내린다.
    """
    high = np.asarray(high, float).copy()
    low  = np.asarray(low,  float).copy()

    # 1) 잠금 값 정의
    LOCK_VALS = {
        0:    0.0,
        1:    0.0,
        4092: 4092.0,
        4095: 4095.0,
    }
    lock_idx = np.array(sorted(LOCK_VALS.keys()), dtype=int)

    # 2) 잠금 인덱스에 원하는 값 강제 세팅
    for j, v in LOCK_VALS.items():
        if 0 <= j < FULL_POINTS:
            high[j] = v

    # 3) low+eps 제약: 잠금 인덱스를 제외한 곳에만 적용
    mask = np.ones_like(high, dtype=bool)
    mask[lock_idx] = False
    high[mask] = np.maximum(high[mask], low[mask] + eps)

    # 4) 중간 구간(2 ~ 4091)에 대해 단조 증가(비감소) 강제
    start_mid = 2
    end_mid   = 4091
    high[start_mid:end_mid+1] = np.maximum.accumulate(high[start_mid:end_mid+1])

    # 5) 0,1 쪽도 음수로 가는 일 방지 (이론상 없지만 방어용)
    high[:2] = np.maximum(high[:2], 0.0)

    # 6) 고계조 쪽이 잠금값(4092, 4095)을 넘지 않도록 clamp
    #    - 0~4091 구간이 4092보다 클 수 없도록
    high[:4092] = np.minimum(high[:4092], high[4092])
    #    - 4092~4095 구간 단조/상한 유지
    high[4092:4095] = np.maximum.accumulate(high[4092:4095])
    high[4092:4096] = np.minimum(high[4092:4096], high[4095])

    # 7) 마지막으로 잠금 인덱스를 한 번 더 덮어써서
    #    중간 과정에서 값이 바뀌지 않았음을 보장
    for j, v in LOCK_VALS.items():
        if 0 <= j < FULL_POINTS:
            high[j] = v

    return high

# ===== 입력 Low LUT 경로 =====
LOW_LUT_CSV = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_low_values_SIN1300.csv"

# ====== 1) High 희소 → 4096 보간 유틸 ======
REQUIRED_SPARSE_COLS = ["Gray", "LUT_j", "R_High", "G_High", "B_High"]

def _ensure_sparse_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_SPARSE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"입력 CSV에 필수 컬럼이 없습니다: {missing}\n"
            f"필수 컬럼: {REQUIRED_SPARSE_COLS}"
        )

def _coerce_sparse_types(df: pd.DataFrame):
    df["Gray"] = pd.to_numeric(df["Gray"], errors="coerce").astype("Int64")
    df["LUT_j"] = pd.to_numeric(df["LUT_j"], errors="coerce")
    for ch in ["R_High","G_High","B_High"]:
        df[ch] = pd.to_numeric(df[ch], errors="coerce")

def _apply_gray254_rule(df: pd.DataFrame):
    # gray=254는 항상 LUT_j=4092 로 매핑
    m254 = df["Gray"] == 254
    if m254.any():
        df.loc[m254, "LUT_j"] = 4092

def _collapse_duplicate_j_keep_last(df: pd.DataFrame, gray_col: str | None = None) -> pd.DataFrame:
    """
    동일 LUT_j가 여러 개면 '마지막 행' 유지.
    sort 키에 Gray 컬럼이 없을 수도 있으므로 옵션 처리
    """
    df2 = df.dropna(subset=["LUT_j"]).copy()
    if gray_col is not None and gray_col in df2.columns:
        df2 = df2.sort_values(["LUT_j", gray_col], kind="mergesort")
    else:
        df2 = df2.sort_values(["LUT_j"], kind="mergesort")
    keep = df2.groupby("LUT_j", as_index=False).tail(1)
    return keep.sort_values("LUT_j").reset_index(drop=True)

def _interp_high_to_4096(df_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    LUT_j축 앵커들을 0..4095로 선형보간해 High 4096포인트 생성
    (gen_random_ref_offset에서 gray12축으로 np.interp 하던 것과 같은 원리)
    """
    x = df_anchor["LUT_j"].to_numpy(dtype=np.float64)
    r = df_anchor["R_High"].to_numpy(dtype=np.float64)
    g = df_anchor["G_High"].to_numpy(dtype=np.float64)
    b = df_anchor["B_High"].to_numpy(dtype=np.float64)
    # 유일성 확보
    x, idx = np.unique(x, return_index=True)
    r = r[idx]; g = g[idx]; b = b[idx]
    full_j = np.arange(0, FULL_POINTS, dtype=np.int32)
    r_full = np.interp(full_j, x, r)
    g_full = np.interp(full_j, x, g)
    b_full = np.interp(full_j, x, b)
    return pd.DataFrame({
        "LUT_j": full_j,
        "R_High_full": r_full.astype(np.float32),
        "G_High_full": g_full.astype(np.float32),
        "B_High_full": b_full.astype(np.float32),
    })

# ====== 2) Low 4096 로드 (12bit) & 형태 표준화 ======
def _load_low_4096(csv_path: str) -> pd.DataFrame:
    """
    기대 포맷 예시:
    - 'LUT_j', 'R_Low', 'G_Low', 'B_Low'  (4096행)
    - LUT_j가 없으면 0..4095 생성
    """
    df = pd.read_csv(csv_path)
    if "LUT_j" not in df.columns:
        df.insert(0, "LUT_j", np.arange(FULL_POINTS, dtype=np.int32))

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_r = pick_col(["R_Low","R","R_low","RChannelLow","RchannelLow"])
    col_g = pick_col(["G_Low","G","G_low","GChannelLow","GchannelLow"])
    col_b = pick_col(["B_Low","B","B_low","BChannelLow","BchannelLow"])
    if not all([col_r, col_g, col_b]):
        raise ValueError(
            "LOW_LUT_CSV에서 R_Low/G_Low/B_Low 열을 찾을 수 없습니다. 열 이름을 확인하세요."
        )

    out = pd.DataFrame({
        "LUT_j": pd.to_numeric(df["LUT_j"], errors="coerce").astype("Int64"),
        "R_Low_full": pd.to_numeric(df[col_r], errors="coerce"),
        "G_Low_full": pd.to_numeric(df[col_g], errors="coerce"),
        "B_Low_full": pd.to_numeric(df[col_b], errors="coerce"),
    })
    if out["LUT_j"].isna().any():
        raise ValueError("LOW_LUT_CSV의 LUT_j에 NaN이 있습니다.")
    if out.shape[0] < FULL_POINTS:
        raise ValueError("LOW_LUT_CSV 행 수가 4096보다 작습니다.")
    out = out.iloc[:FULL_POINTS].reset_index(drop=True)

    # Low도 혹시 모를 비단조 구간을 정리
    out["R_Low_full"] = _enforce_monotone(out["R_Low_full"].to_numpy(float))
    out["G_Low_full"] = _enforce_monotone(out["G_Low_full"].to_numpy(float))
    out["B_Low_full"] = _enforce_monotone(out["B_Low_full"].to_numpy(float))

    return out

# ====== 3) High 4096 생성 ======
def _compute_high_full_4096_from_sparse(sparse_high_csv: str) -> pd.DataFrame:
    df_s = pd.read_csv(sparse_high_csv)
    _ensure_sparse_columns(df_s)
    _coerce_sparse_types(df_s)
    _apply_gray254_rule(df_s)

    # LUT_j축 보간용 앵커 구성 (중복 LUT_j는 마지막 행 유지)
    df_anchor = _collapse_duplicate_j_keep_last(
        df_s[["LUT_j","R_High","G_High","B_High"]], gray_col=None
    )
    return _interp_high_to_4096(df_anchor)  # (4096행 DataFrame)

# ====== 4) 최종 4096 테이블 구성 ======
def build_full_4096_table(sparse_high_csv: str, low_4096_csv: str) -> pd.DataFrame:
    """
    LUT_j=0..4095 전 구간 4096행 테이블 생성
      (출력 컬럼은 저장 시 6개만 선택)
    - Low: 원본 4096 (단조 보정 포함)
    - High: 희소를 LUT_j축으로 보간한 4096 + Low+eps + 단조 + lock 인덱스 제약
    """
    df_high4096 = _compute_high_full_4096_from_sparse(sparse_high_csv)
    df_low4096  = _load_low_4096(low_4096_csv)

    # numpy 배열로 꺼내기
    Rl = df_low4096["R_Low_full"].to_numpy(float)
    Gl = df_low4096["G_Low_full"].to_numpy(float)
    Bl = df_low4096["B_Low_full"].to_numpy(float)

    Rh = df_high4096["R_High_full"].to_numpy(float)
    Gh = df_high4096["G_High_full"].to_numpy(float)
    Bh = df_high4096["B_High_full"].to_numpy(float)

    # gen_random_ref_offset.py와 유사하게:
    # - High ≥ Low + eps
    # - 단조 증가
    # - j=0,1,4092,4095 잠금
    Rh = _enforce_with_locks_12bit(Rh, Rl, EPS_HIGH_OVER_LOW)
    Gh = _enforce_with_locks_12bit(Gh, Gl, EPS_HIGH_OVER_LOW)
    Bh = _enforce_with_locks_12bit(Bh, Bl, EPS_HIGH_OVER_LOW)

    df = pd.DataFrame({
        "LUT_j": np.arange(FULL_POINTS, dtype=np.int32),
        "R_Low":  _clip_round_12bit(Rl),
        "R_High": _clip_round_12bit(Rh),
        "G_Low":  _clip_round_12bit(Gl),
        "G_High": _clip_round_12bit(Gh),
        "B_Low":  _clip_round_12bit(Bl),
        "B_High": _clip_round_12bit(Bh),
    })
    return df

# ====== 5) 메인 플로우 ======
def main():
    if use_gui:
        root = tk.Tk(); root.withdraw()

    # 1) 희소 High CSV 선택 (필수)
    if use_gui:
        sparse_csv = askopenfilename(
            title="Gray-LUT_j-High(희소) CSV 선택",
            filetypes=[("CSV Files","*.csv"),("All Files","*.*")]
        )
        if not sparse_csv:
            print("@INFO: 입력 파일을 선택하지 않아 종료합니다.")
            return
    else:
        sparse_csv = input("희소 High CSV 경로: ").strip()
        if not sparse_csv:
            print("@INFO: 입력 파일 경로가 비었습니다.")
            return

    # 2) 4096 풀 테이블 생성
    df_full4096 = build_full_4096_table(
        sparse_high_csv=sparse_csv,
        low_4096_csv=LOW_LUT_CSV
    )

    # 3) 임시 파일로 저장 (요청: 최종 6컬럼만)
    with tempfile.NamedTemporaryFile(delete=False, suffix="_LUT_full4096.csv") as tmp:
        tmp_path = tmp.name

    cols6 = ["R_Low","R_High","G_Low","G_High","B_Low","B_High"]
    df_full4096.to_csv(tmp_path, index=False, encoding="utf-8-sig", columns=cols6)
    print(f"[OK] 4096-포인트 LUT CSV(임시) 저장: {tmp_path}")

    # 4) Windows에서 바로 열기 시도
    try:
        os.startfile(tmp_path)  # type: ignore[attr-defined]
    except Exception:
        pass

if __name__ == "__main__":
    main()