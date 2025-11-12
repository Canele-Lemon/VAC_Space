# interpolate_lut_j_to_4096.py
# - 희소 High(Gray, LUT_j, R_High, G_High, B_High) → LUT_j 축 0..4095 보간
# - Low(12bit, 4096포인트)는 보간/정규화 없이 '있는 그대로'
# - LUT_j 매핑으로 256 그레이 테이블 생성(Gray 0..255)
# - 중간 CSV 저장 없이 바로 JSON 생성
# - JSON 파일명은 입력 희소 CSV 파일명에서 파생

import os
import sys
import numpy as np
import pandas as pd

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
        raise ValueError(
            f"입력 CSV에 필수 컬럼이 없습니다: {missing}\n"
            f"필수 컬럼: {REQUIRED_SPARSE_COLS}"
        )

def _coerce_sparse_types(df: pd.DataFrame):
    df["Gray"] = pd.to_numeric(df["Gray"], errors="coerce").astype("Int64")
    df["LUT_j"] = pd.to_numeric(df["LUT_j"], errors="coerce")
    for ch in ["R_High", "G_High", "B_High"]:
        df[ch] = pd.to_numeric(df[ch], errors="coerce")

def _apply_gray254_rule(df: pd.DataFrame):
    # 254 gray는 항상 LUT_j=4092 (요구사항)
    m254 = df["Gray"] == 254
    if m254.any():
        df.loc[m254, "LUT_j"] = 4092

def _collapse_duplicate_j_keep_last(df: pd.DataFrame, gray_col: str | None = None) -> pd.DataFrame:
    """
    동일 LUT_j가 여러 개면 '마지막 행'을 유지.
    gray_col이 주어지면 (LUT_j, gray_col) 기준 안정 정렬 후 tail(1)로 '마지막' 정의.
    """
    # 입력 df는 최소 ["LUT_j", "R_High", "G_High", "B_High"] 여야 함.
    # gray_col이 없으면 LUT_j만으로 tail(1)
    if gray_col is not None and gray_col in df.columns:
        df2 = df.dropna(subset=["LUT_j"]).sort_values(["LUT_j", gray_col], kind="mergesort")
    else:
        df2 = df.dropna(subset=["LUT_j"]).sort_values(["LUT_j"], kind="mergesort")
    keep = df2.groupby("LUT_j", as_index=False).tail(1)
    return keep.sort_values("LUT_j").reset_index(drop=True)

def _interp_to_4096(df_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    LUT_j축 앵커들을 0..4095로 선형 보간.
    입력 컬럼: ["LUT_j","R_High","G_High","B_High"]
    """
    x = df_anchor["LUT_j"].to_numpy(dtype=np.float64)
    r = df_anchor["R_High"].to_numpy(dtype=np.float64)
    g = df_anchor["G_High"].to_numpy(dtype=np.float64)
    b = df_anchor["B_High"].to_numpy(dtype=np.float64)

    # 유일성 확보
    x, idx = np.unique(x, return_index=True)
    r = r[idx]; g = g[idx]; b = b[idx]

    if x.size < 2:
        raise ValueError("보간을 위해서는 서로 다른 LUT_j 앵커가 최소 2개 이상 필요합니다.")

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

# ====== 2) Low 4096 로드 (12bit 원본 그대로 사용) ======
def _load_low_4096(csv_path: str) -> pd.DataFrame:
    """
    기대 포맷(열 이름은 방어적으로 매핑):
    - 'LUT_j', 'R_Low', 'G_Low', 'B_Low'  (4096행)
      또는 'R','G','B' 등 변형 컬럼명도 일부 허용
    어떤 보간/정규화/스케일링도 하지 않음 (원본 유지)
    """
    df = pd.read_csv(csv_path)

    if "LUT_j" not in df.columns:
        df.insert(0, "LUT_j", np.arange(4096, dtype=np.int32))

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_r = pick_col(["R_Low", "R", "R_low", "RChannelLow", "RchannelLow"])
    col_g = pick_col(["G_Low", "G", "G_low", "GChannelLow", "GchannelLow"])
    col_b = pick_col(["B_Low", "B", "B_low", "BChannelLow", "BchannelLow"])
    if not all([col_r, col_g, col_b]):
        raise ValueError("LOW_LUT_CSV에서 R_Low/G_Low/B_Low 열을 찾을 수 없습니다.")

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

# ====== 3) 256 그레이 테이블 생성 ======
def build_gray_table(sparse_high_csv: str, low_4096_csv: str) -> pd.DataFrame:
    # 희소 High 로드
    df_s = pd.read_csv(sparse_high_csv)
    _ensure_sparse_columns(df_s)
    _coerce_sparse_types(df_s)
    _apply_gray254_rule(df_s)

    # 256행만 남기고 기본 정렬
    df_s = df_s.dropna(subset=["Gray", "LUT_j"]).copy()
    df_s["Gray"] = df_s["Gray"].astype(int)
    df_s = df_s.sort_values("Gray")

    # High 보간(0..4095)
    df_anchor = _collapse_duplicate_j_keep_last(
        df_s[["LUT_j", "R_High", "G_High", "B_High"]], gray_col=None
    )
    df_high4096 = _interp_to_4096(df_anchor).set_index("LUT_j")

    # Low 4096 원본 로드
    df_low4096 = _load_low_4096(low_4096_csv).set_index("LUT_j")

    # 그레이별 매핑 j로 샘플링
    rows = []
    for _, row in df_s.iterrows():
        g = int(row["Gray"])
        j = int(row["LUT_j"])
        j = max(0, min(4095, j))

        # Low: 원본 값 그대로
        R_low = df_low4096.at[j, "R_Low_full"]
        G_low = df_low4096.at[j, "G_Low_full"]
        B_low = df_low4096.at[j, "B_Low_full"]

        # High: 보간된 4096에서 동일 j
        R_hih = df_high4096.at[j, "R_High_full"]
        G_hih = df_high4096.at[j, "G_High_full"]
        B_hih = df_high4096.at[j, "B_High_full"]

        rows.append({
            "GrayLevel_window": g,
            "R_Low":  int(round(R_low)),
            "R_High": int(round(R_hih)),
            "G_Low":  int(round(G_low)),
            "G_High": int(round(G_hih)),
            "B_Low":  int(round(B_low)),
            "B_High": int(round(B_hih)),
        })

    df_out256 = pd.DataFrame(rows).sort_values("GrayLevel_window").reset_index(drop=True)
    return df_out256

# ====== 4) JSON 포맷터 ======
def write_default_data(file):
    default_data = """{																					
"DRV_valc_major_ctrl"	:	[	0,	1	],																
"DRV_valc_pattern_ctrl_0"	:	[	5,	1	],																
"DRV_valc_pattern_ctrl_1"	:	[	[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	]      	 ],
"DRV_valc_sat_ctrl"	:	[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],		
"DRV_valc_hpf_ctrl_0"	:	[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],		
"DRV_valc_hpf_ctrl_1"	:		1,																		
"""
    file.write(default_data)

def write_LUT_data(file, df_gray: pd.DataFrame):
    """
    df_gray: build_gray_table() 결과 (256행 x 7열)
    JSON은 16x16 블록으로 포맷팅
    """
    channels = {
        "RchannelLow": 'R_Low',
        "RchannelHigh": 'R_High',
        "GchannelLow": 'G_Low',
        "GchannelHigh": 'G_High',
        "BchannelLow": 'B_Low',
        "BchannelHigh": 'B_High'
    }

    for i, (channel_name, col) in enumerate(channels.items()):
        file.write(f'"{channel_name}"\t:\t[\t')
        data = df_gray[col].to_numpy()
        if data.size != 256:
            raise ValueError(f"{col} 길이가 256이 아닙니다: {data.size}")
        # 16x16으로 출력
        reshaped_data = data.reshape(16, 16)

        for row_index, row in enumerate(reshaped_data):
            formatted_row = ',\t'.join(map(lambda x: str(int(x)), row))
            if row_index == 0:
                file.write(f'{formatted_row},\n')
            elif row_index == reshaped_data.shape[0] - 1:
                file.write(f'\t\t\t{formatted_row}')
            else:
                file.write(f'\t\t\t{formatted_row},\n')

        if i == len(channels) - 1:
            file.write("\t]\n")
        else:
            file.write("\t],\n")
    file.write("}")

# ====== 5) 메인 플로우 ======
def main():
    if use_gui:
        root = tk.Tk(); root.withdraw()

    # 1) 희소 High CSV 선택
    if use_gui:
        sparse_csv = askopenfilename(
            title="Gray-LUT_j-High(희소) CSV 선택",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not sparse_csv:
            print("@INFO: 입력 파일을 선택하지 않아 종료합니다."); return
    else:
        sparse_csv = input("희소 High CSV 경로: ").strip()
        if not sparse_csv:
            print("@INFO: 입력 파일을 제공하지 않아 종료합니다."); return

    # 2) 256 그레이 테이블 생성 (중간 CSV 저장 없음)
    df_gray = build_gray_table(sparse_high_csv=sparse_csv, low_4096_csv=LOW_LUT_CSV)
    print("[OK] 256 그레이 LUT 메모리상 생성 완료")

    # 3) JSON 저장 경로 (파일명은 입력 희소 CSV 기반)
    base_name = os.path.splitext(os.path.basename(sparse_csv))[0]
    default_json_name = f"LUT_DGA_{base_name}.json"

    if use_gui:
        out_json_path = asksaveasfilename(
            title="JSON 저장",
            defaultextension=".json",
            initialfile=default_json_name,
            filetypes=[("JSON Files", "*.json")]
        )
        if not out_json_path:
            # 사용자가 취소하면 입력파일과 같은 폴더에 기본 파일명으로 저장
            out_json_path = os.path.join(os.path.dirname(sparse_csv), default_json_name)
    else:
        out_json_path = os.path.join(os.path.dirname(sparse_csv), default_json_name)

    # 4) JSON 생성
    with open(out_json_path, 'w', encoding='utf-8') as f:
        write_default_data(f)
        write_LUT_data(f, df_gray)

    print(f"[OK] JSON 생성 완료: {out_json_path}")

    # Windows에서 열기 시도
    try:
        os.startfile(out_json_path)  # type: ignore[attr-defined]
    except Exception:
        pass

    if use_gui:
        messagebox.showinfo("완료", f"JSON 생성 완료\n\nJSON: {out_json_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        if use_gui:
            messagebox.showerror("오류", str(e))