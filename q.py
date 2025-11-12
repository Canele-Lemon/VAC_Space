# interpolate_lut_j_to_4096.py

import os
import sys
import numpy as np
import pandas as pd
import tempfile

# ----- GUI -----
try:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter import messagebox
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

# ====== 3) Gray→j 보간(단조/규칙 반영) + 256 그레이 테이블 ======
def _gray_to_j_dense(df_gray_lutj: pd.DataFrame) -> np.ndarray:
    """
    Gray, LUT_j 앵커로 Gray→j를 보간해서 256 길이의 정수 j 배열 생성.
    - 단조 증가(비감소) 강제
    - gray=254 → 4092 (규칙 유지)
    - gray=255 → 4095 (끝점 강제)
    """
    # 같은 Gray가 여러 번이면 마지막 행 유지
    df = df_gray_lutj.dropna(subset=["Gray","LUT_j"]).copy()
    df["Gray"] = df["Gray"].astype(int)
    df = df.sort_values(["Gray"], kind="mergesort")
    df = df.groupby("Gray", as_index=False).tail(1)

    gray_anchor = df["Gray"].to_numpy(dtype=np.float64)
    j_anchor    = df["LUT_j"].to_numpy(dtype=np.float64)

    if gray_anchor.size < 2:
        raise ValueError("Gray→LUT_j 보간을 위해서는 서로 다른 Gray 앵커가 최소 2개 필요합니다.")

    full_gray = np.arange(256, dtype=np.float64)
    lut_j_dense = np.interp(full_gray, gray_anchor, j_anchor)  # Gray축 보간된 j(g)

    # 정수화 + 경계 클립
    lut_j_dense = np.rint(lut_j_dense).astype(np.int32)
    lut_j_dense = np.clip(lut_j_dense, 0, 4095)

    # 단조 증가(비감소) 강제
    lut_j_dense = np.maximum.accumulate(lut_j_dense)

    # 규칙 반영
    lut_j_dense[254] = 4092          # 규칙 유지
    lut_j_dense[255] = 4095          # 끝점 강제
    lut_j_dense[:255] = np.minimum(lut_j_dense[:255], 4095)

    return lut_j_dense

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

def build_gray_table(sparse_high_csv: str, low_4096_csv: str) -> pd.DataFrame:
    """
    (요구1) 256개 Gray에 대해 보간된 j(g)를 사용해 Low/High를 샘플링한 256행 테이블
    """
    # 희소 High 로드
    df_s = pd.read_csv(sparse_high_csv)
    _ensure_sparse_columns(df_s)
    _coerce_sparse_types(df_s)
    _apply_gray254_rule(df_s)

    # High 4096 보간
    df_high4096 = _compute_high_full_4096_from_sparse(sparse_high_csv).set_index("LUT_j")
    # Low 4096 로드
    df_low4096 = _load_low_4096(low_4096_csv).set_index("LUT_j")

    # Gray→j(g) 보간(단조/규칙 반영)
    lut_j_dense = _gray_to_j_dense(df_s[["Gray","LUT_j"]])

    # 256행 테이블 생성
    rows = []
    for g in range(256):
        j = int(lut_j_dense[g])

        R_low = df_low4096.at[j, "R_Low_full"]
        G_low = df_low4096.at[j, "G_Low_full"]
        B_low = df_low4096.at[j, "B_Low_full"]

        R_hih = df_high4096.at[j, "R_High_full"]
        G_hih = df_high4096.at[j, "G_High_full"]
        B_hih = df_high4096.at[j, "B_High_full"]

        rows.append({
            "GrayLevel_window": g,
            "LUT_j": j,
            "R_Low":  int(round(R_low)),
            "R_High": int(round(R_hih)),
            "G_Low":  int(round(G_low)),
            "G_High": int(round(G_hih)),
            "B_Low":  int(round(B_low)),
            "B_High": int(round(B_hih)),
        })

    df_out256 = pd.DataFrame(rows)
    return df_out256

def build_full_4096_table(sparse_high_csv: str, low_4096_csv: str) -> pd.DataFrame:
    """
    (요구2) LUT_j=0..4095 전 구간 4096행 테이블 생성
      컬럼: LUT_j, R_Low, R_High, G_Low, G_High, B_Low, B_High
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

# ====== 4) 메인 플로우 ======
def main():
    if use_gui:
        root = tk.Tk(); root.withdraw()

    # 1) 희소 High CSV 선택
    if use_gui:
        sparse_csv = askopenfilename(title="Gray-LUT_j-High(희소) CSV 선택",
                                     filetypes=[("CSV Files","*.csv"),("All Files","*.*")])
        if not sparse_csv:
            print("@INFO: 입력 파일을 선택하지 않아 종료합니다."); return
    else:
        sparse_csv = input("희소 High CSV 경로: ").strip()

    # 2) 256 그레이 테이블 생성 + 저장(선택)
    df_gray = build_gray_table(sparse_high_csv=sparse_csv, low_4096_csv=LOW_LUT_CSV)

    if use_gui:
        save_csv_256 = asksaveasfilename(title="병합 LUT(256) CSV 저장",
                                         defaultextension=".csv",
                                         initialfile="LUT_gray_merged_256.csv",
                                         filetypes=[("CSV Files","*.csv")])
        if not save_csv_256:
            base, ext = os.path.splitext(sparse_csv)
            save_csv_256 = f"{base}_merged_256.csv"
    else:
        base, ext = os.path.splitext(sparse_csv)
        save_csv_256 = f"{base}_merged_256.csv"

    df_gray.to_csv(save_csv_256, index=False, encoding="utf-8-sig")
    print(f"[OK] 256 그레이 LUT CSV 저장: {save_csv_256}")

    # 3) 4096 풀 테이블 생성
    df_full4096 = build_full_4096_table(sparse_high_csv=sparse_csv, low_4096_csv=LOW_LUT_CSV)

    # 4) 임시 파일로 저장 후 바로 열기
    with tempfile.NamedTemporaryFile(delete=False, suffix="_LUT_full4096.csv") as tmp:
        tmp_path = tmp.name
    df_full4096.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 4096-포인트 LUT CSV(임시) 저장: {tmp_path}")

    # Windows에서 열기 시도
    try:
        os.startfile(tmp_path)  # type: ignore[attr-defined]
    except Exception:
        pass

    if use_gui:
        messagebox.showinfo("완료",
                            f"CSV 생성 완료\n\n256 Gray: {save_csv_256}\n4096 LUT(임시): {tmp_path}")

if __name__ == "__main__":
    main()

위 코드를 다음과 같이 수정해주세요:
1. LUT_gray_merged_256.csv 파일이 왜 저장되는지 모르겠는데 filedialog 나오면서 저장하는 기능는 없애주세요. 저장 안해도 됩니다.
2. 최종 R_Low	R_High	G_Low	G_High	B_Low	B_High 4096 데이터 csv 임시파일만 뜨게 해 주세요.
3. 생성 완료 window 안내창 안뜨게 해 주세요.
