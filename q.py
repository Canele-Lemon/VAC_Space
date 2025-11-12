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

# ===== 사용자가 제공한 LOW LUT 경로 (12bit, 4096포인트) =====
LOW_LUT_CSV = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_low_values_SIN1300.csv"

# ====== 1) High 희소 → 4096 보간 유틸 ======
REQUIRED_SPARSE_COLS = ["Gray", "LUT_j", "R_High", "G_High", "B_High"]

def _ensure_sparse_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_SPARSE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"입력 CSV에 필수 컬럼이 없습니다: {missing}\n"
                         f"필수 컬럼: {REQUIRED_SPARSE_COLS}")

def _coerce_sparse_types(df: pd.DataFrame):
    df["Gray"] = pd.to_numeric(df["Gray"], errors="coerce").astype("Int64")
    df["LUT_j"] = pd.to_numeric(df["LUT_j"], errors="coerce")
    for ch in ["R_High","G_High","B_High"]:
        df[ch] = pd.to_numeric(df[ch], errors="coerce")

def _apply_gray254_rule(df: pd.DataFrame):
    # gray=254는 항상 4092
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
    """
    x = df_anchor["LUT_j"].to_numpy(dtype=np.float64)
    r = df_anchor["R_High"].to_numpy(dtype=np.float64)
    g = df_anchor["G_High"].to_numpy(dtype=np.float64)
    b = df_anchor["B_High"].to_numpy(dtype=np.float64)
    # 유일성 확보
    x, idx = np.unique(x, return_index=True)
    r = r[idx]; g = g[idx]; b = b[idx]
    full_j = np.arange(0, 4096, dtype=np.int32)
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
        df.insert(0, "LUT_j", np.arange(4096, dtype=np.int32))

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_r = pick_col(["R_Low","R","R_low","RChannelLow","RchannelLow"])
    col_g = pick_col(["G_Low","G","G_low","GChannelLow","GchannelLow"])
    col_b = pick_col(["B_Low","B","B_low","BChannelLow","BchannelLow"])
    if not all([col_r, col_g, col_b]):
        raise ValueError("LOW_LUT_CSV에서 R_Low/G_Low/B_Low 열을 찾을 수 없습니다. 열 이름을 확인하세요.")

    out = pd.DataFrame({
        "LUT_j": pd.to_numeric(df["LUT_j"], errors="coerce").astype("Int64"),
        "R_Low_full": pd.to_numeric(df[col_r], errors="coerce"),
        "G_Low_full": pd.to_numeric(df[col_g], errors="coerce"),
        "B_Low_full": pd.to_numeric(df[col_b], errors="coerce"),
    })
    if out["LUT_j"].isna().any():
        raise ValueError("LOW_LUT_CSV의 LUT_j에 NaN이 있습니다.")
    if out.shape[0] < 4096:
        raise ValueError("LOW_LUT_CSV 행 수가 4096보다 작습니다.")
    return out.iloc[:4096].reset_index(drop=True)

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
      (출력 컬럼은 나중에 저장 시 6개만 선택)
    - Low: 원본 4096 그대로
    - High: 희소를 LUT_j축으로 보간한 4096
    """
    df_high4096 = _compute_high_full_4096_from_sparse(sparse_high_csv)
    df_low4096  = _load_low_4096(low_4096_csv)

    df = pd.DataFrame({
        "LUT_j": np.arange(4096, dtype=np.int32),
        "R_Low":  df_low4096["R_Low_full"].to_numpy(dtype=np.float64),
        "R_High": df_high4096["R_High_full"].to_numpy(dtype=np.float64),
        "G_Low":  df_low4096["G_Low_full"].to_numpy(dtype=np.float64),
        "G_High": df_high4096["G_High_full"].to_numpy(dtype=np.float64),
        "B_Low":  df_low4096["B_Low_full"].to_numpy(dtype=np.float64),
        "B_High": df_high4096["B_High_full"].to_numpy(dtype=np.float64),
    })
    # 정수화(필요 시)
    for c in ["R_Low","R_High","G_Low","G_High","B_Low","B_High"]:
        df[c] = np.rint(df[c]).astype(int)
    return df

# ====== 5) 메인 플로우 ======
def main():
    if use_gui:
        root = tk.Tk(); root.withdraw()

    # 1) 희소 High CSV 선택 (필수)
    if use_gui:
        sparse_csv = askopenfilename(title="Gray-LUT_j-High(희소) CSV 선택",
                                     filetypes=[("CSV Files","*.csv"),("All Files","*.*")])
        if not sparse_csv:
            print("@INFO: 입력 파일을 선택하지 않아 종료합니다.")
            return
    else:
        sparse_csv = input("희소 High CSV 경로: ").strip()
        if not sparse_csv:
            print("@INFO: 입력 파일 경로가 비었습니다.")
            return

    # 2) 4096 풀 테이블 생성
    df_full4096 = build_full_4096_table(sparse_high_csv=sparse_csv, low_4096_csv=LOW_LUT_CSV)

    # 3) 임시 파일로 저장 (요청: 최종 6컬럼만, LUT_j 제외)
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