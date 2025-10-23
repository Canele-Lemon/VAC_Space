# VAC_dataset.py
import sys
import torch
import os
import json
import pandas as pd
import numpy as np
import tempfile, webbrowser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from src.prepare_input import VACInputBuilder
from src.prepare_output import VACOutputBuilder

# ---------------------------------------------------------
# 상수/유틸
# ---------------------------------------------------------
_PATTERN_LIST = ['W', 'R', 'G', 'B']
_MACBETH_LIST = [
    "Red","Green","Blue","Cyan","Magenta","Yellow",
    "White","Gray","Darkskin","Lightskin","Asian","Western"
]
def _onehot(idx: int, size: int) -> np.ndarray:
    """
    범주형 인덱스를 원-핫 벡터로 변환.

    Parameters
    ----------
    idx : int
        활성화할 인덱스 (0 ~ size-1), 범위를 벗어나면 전체 0
    size : int
        벡터 길이

    Returns
    -------
    np.ndarray, shape (size,)
    """
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

class VACDataset(Dataset):
    """
    VACDataset: VACInputBuilder와 VACOutputBuilder로부터
    PK 리스트 기반의 구조화된 X/Y를 수집하고
    모델 유형별로 학습용 (X, y) 행렬/벡터를 조립하는 클래스
    """

    def __init__(self, pk_list):
        """
        초기화
        :param pk_list: 사용할 PK 번호 리스트
        """
        self.pk_list = list(pk_list)
        self.samples = []   # 원본 구조 보관
        self._collect()

    # -----------------------------------------------------
    # 원본(X/Y) 수집: dict 그대로 보관
    # -----------------------------------------------------
    def _collect(self):
        for pk in self.pk_list:
            x_builder = VACInputBuilder(pk)
            y_builder = VACOutputBuilder(pk)
            X = x_builder.prepare_X0()   # {"lut": {...}, "meta": {...}}
            Y = y_builder.prepare_Y(y1_patterns=('W',))    # {"Y0": {...}, "Y1": {...}, "Y2": {...}}
            self.samples.append({"pk": pk, "X": X, "Y": Y})

    # -----------------------------------------------------
    # 내부: 특정 gray에서 피처 벡터 만들기
    # -----------------------------------------------------
    def _build_features_for_gray(self, X_dict, gray: int, add_pattern: str | None = None,
                                 channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High')) -> np.ndarray:
        """
        특정 gray(0~255)에서 6채널 LUT 값 + 메타 + gray_norm + (옵션)패턴 원핫을 붙여 피처 구성.

        feature = [
            R_Low[g], R_High[g], G_Low[g], G_High[g], B_Low[g], B_High[g],
            panel_onehot..., frame_rate, model_year,
            gray/255,
            (opt) pattern_onehot(4)
        ]
        """
        lut = X_dict["lut"]
        meta = X_dict["meta"]

        row = []
        for ch in channels:
            row.append(float(lut[ch][gray]))

        # 메타
        panel_vec = np.asarray(meta["panel_maker"], dtype=np.float32).tolist()
        row.extend(panel_vec)
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # 그레이 정규화 인덱스
        row.append(gray / 255.0)

        # (옵션) 패턴 원핫
        if add_pattern is not None:
            pat_idx = _PATTERN_LIST.index(add_pattern) if add_pattern in _PATTERN_LIST else -1
            row.extend(_onehot(pat_idx, len(_PATTERN_LIST)).tolist())

        return np.asarray(row, dtype=np.float32)
    
    def _build_features_for_segment(self, X_dict, g: int, add_pattern: str | None = None,
                                    low_only: bool = False) -> np.ndarray:
        """
        구간(g -> g+1) 예측용 피처:
        [ LUT@g , LUT@(g+1) , (LUT@diff=g+1 - g) , meta(centered_gray_norm)/패턴 ]
        - low_only=True면 LUT 채널 중 R/G/B Low만 사용 (R_Low, G_Low, B_Low)
        - gray_norm은 세그먼트 중심값 (g+0.5)/255 로 교체하여 사용
        """
        # 안전 범위
        if not (0 <= g < 255):
            raise ValueError(f"segment g must be in [0, 254], got g={g}")

        # g, g+1 시점 피처 (한 행씩)
        f_g  = self._build_features_for_gray(X_dict, g,   add_pattern=add_pattern)
        f_g1 = self._build_features_for_gray(X_dict, g+1, add_pattern=add_pattern)

        # --- 1) LUT 부분 분리 및 (옵션) Low-only ---
        # 앞 6개가 LUT: [R_Low, R_High, G_Low, G_High, B_Low, B_High]
        lut_g  = f_g[:6].copy()
        lut_g1 = f_g1[:6].copy()
        if low_only:
            lut_idx = [0, 2, 4]  # Low만
            lut_g   = lut_g[lut_idx]
            lut_g1  = lut_g1[lut_idx]

        lut_diff = lut_g1 - lut_g  # 구간 차분

        # --- 2) meta tail 구성 (panel_onehot, frame_rate, model_year, gray_norm, pattern_onehot)
        # f_g 기준 tail 사용 (pattern/onehot 포함)
        meta_tail = f_g[6:].copy()

        # --- 3) gray_norm을 세그먼트 중심값으로 교체 ---
        # panel_onehot 길이(K)는 meta에서 확인
        K = len(X_dict["meta"]["panel_maker"])  # one-hot 길이
        # tail 배열 내에서 gray_norm의 위치:
        # tail = [panel_onehot(K), frame_rate(1), model_year(1), gray_norm(1), pattern_onehot(4)]
        idx_gray_in_tail = K + 2
        if idx_gray_in_tail >= meta_tail.shape[0]:
            # 방어: 스키마가 바뀐 경우를 대비
            raise IndexError("gray_norm index is out of range in meta_tail; "
                            "check _build_features_for_gray feature layout.")
        # 세그먼트 중심값
        meta_tail[idx_gray_in_tail] = (g + 0.5) / 255.0

        # --- 4) 최종 피처 결합 ---
        feat = np.concatenate([lut_g, lut_g1, lut_diff, meta_tail]).astype(np.float32)
        return feat

    # -----------------------------------------------------
    # 1) 멀티타깃 일괄학습용 (전체 플랫)
    #    X: LUT(6*256) + meta
    #    Y: Y0(4*3*256) + Y1(4*255) + Y2(12)
    # -----------------------------------------------------
    def build_multitarget_flat(self, include=('Y0', 'Y1', 'Y2')):
        """
        멀티타깃 일괄학습(벡터 플랫) 데이터셋 생성.

        Returns
        -------
        X_mat : np.ndarray, shape (N, Dx)
            N = 샘플 수 (PK 개수),
            Dx = 6*256 + |panel_onehot| + 2(frame_rate, model_year)
        Y_mat : np.ndarray, shape (N, Dy)
            Dy = sum(include):
                - Y0: 4패턴 * 3컴포넌트 * 256 = 3072
                - Y1: 'W' 패턴 * 255 = 255
                - Y2: 12
              예) include=('Y0','Y1','Y2') → 3339
        """
        X_rows, Y_rows = [], []
        for s in self.samples:
            Xd = s["X"]; Yd = s["Y"]
            # X 플랫
            lut = Xd["lut"]; meta = Xd["meta"]
            x_flat = np.concatenate([
                lut['R_Low'], lut['R_High'],
                lut['G_Low'], lut['G_High'],
                lut['B_Low'], lut['B_High'],
                meta['panel_maker'].astype(np.float32),
                np.array([meta['frame_rate'], meta['model_year']], dtype=np.float32)
            ]).astype(np.float32)

            # Y 플랫
            y_parts = []
            if 'Y0' in include:
                for p in _PATTERN_LIST:
                    y_parts.append(np.nan_to_num(Yd['Y0'][p]['Gamma'], nan=0.0, posinf=0.0, neginf=0.0))
                    y_parts.append(np.nan_to_num(Yd['Y0'][p]['Cx'],    nan=0.0, posinf=0.0, neginf=0.0))
                    y_parts.append(np.nan_to_num(Yd['Y0'][p]['Cy'],    nan=0.0, posinf=0.0, neginf=0.0))
            if 'Y1' in include:
                y_parts.append(Yd['Y1']['W'])
            if 'Y2' in include:
                y_parts.append(np.array([Yd['Y2'][m] for m in _MACBETH_LIST], dtype=np.float32))

            y_flat = np.concatenate(y_parts).astype(np.float32)
            X_rows.append(x_flat)
            Y_rows.append(y_flat)

        X_mat = np.vstack(X_rows).astype(np.float32)
        Y_mat = np.vstack(Y_rows).astype(np.float32)
        return X_mat, Y_mat

    # -----------------------------------------------------
    # 2) Y0(계조별 dGamma/dCx/dCy) 회귀 (선형 추세 등)
    #    행 단위: (pk, pattern, gray)
    # -----------------------------------------------------
    # def build_per_gray_y0(self, component='Gamma', patterns=('W','R','G','B')):
    #     """
    #     Y0(계조별 Gamma/Cx/Cy) 단일 스칼라 회귀용 데이터셋.

    #     Parameters
    #     ----------
    #     component : {'Gamma','Cx','Cy'}
    #     patterns : tuple[str]

    #     Returns
    #     -------
    #     X_mat : np.ndarray, shape (N * len(patterns) * 256, Dx)
    #         행 단위 = (pk, pattern, gray)
    #     y_vec : np.ndarray, shape (N * len(patterns) * 256,)
    #         타깃 스칼라 (선택한 component 값)
    #     """
        # assert component in ('Gamma', 'Cx', 'Cy')
        # X_rows, y_vals = [], []
        # for s in self.samples:
        #     Xd = s["X"]; Yd = s["Y"]
        #     for p in patterns:
        #         y_vec = Yd['Y0'][p][component]  # (256,)
        #         for g in range(256):
        #             y_val = y_vec[g]
        #             if not np.isfinite(y_val):     # NaN/inf는 스킵
        #                 continue
        #             X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
        #             y_vals.append(float(y_val))
        # X_mat = np.vstack(X_rows).astype(np.float32)
        # y_vec = np.asarray(y_vals, dtype=np.float32)
        # return X_mat, y_vec
    def build_per_gray_y0(self, component='Gamma', patterns=('W','R','G','B')):
        assert component in ('Gamma', 'Cx', 'Cy')
        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk = s["pk"]
            Xd = s["X"]; Yd = s["Y"]
            for p in patterns:
                y_vec = Yd['Y0'][p][component]  # (256,)
                for g in range(256):
                    y_val = y_vec[g]
                    if not np.isfinite(y_val):   # NaN/inf는 스킵
                        continue
                    X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
                    y_vals.append(float(y_val))
                    groups.append(pk)            # ← 유지된 행에 대해 pk를 같이 쌓기

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

    # -----------------------------------------------------
    # 3) Y1(측면 slope) 회귀 (gray 0~254)
    #    행 단위: (pk, pattern, gray_segment)
    # -----------------------------------------------------
    def build_per_gray_y1(self, patterns=('W',), use_segment_features=True, low_only=True):
        """
        Y1(측면 중계조 slope) 단일 스칼라 회귀용 데이터셋.

        Parameters
        ----------
        patterns : tuple[str]

        Returns
        -------
        X_mat : np.ndarray, shape (N * len(patterns) * 255, Dx)
            행 단위 = (pk, pattern, gray_segment)
        y_vec : np.ndarray, shape (N * len(patterns) * 255,)
            slope 값
        """
        X_rows, y_vals = [], []
        for s in self.samples:
            Xd = s["X"]; Yd = s["Y"]
            for p in patterns:
                slope = Yd['Y1'][p]  # (255,)
                for g in range(255):
                    if use_segment_features:
                        X_rows.append(self._build_features_for_segment(Xd, g, add_pattern=p, low_only=low_only))
                    else:
                        # g 시점만
                        X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
                    y_vals.append(float(slope[g]))
        X_mat = np.vstack(X_rows).astype(np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        return X_mat, y_vec

    # -----------------------------------------------------
    # 4) Y2(Δu'v') 회귀: Macbeth 12패치 스칼라
    #    행 단위: (pk, macbeth_patch)
    # -----------------------------------------------------
    def build_y2_macbeth(self, use_lut_summary: bool = True):
        """
        Y2(Δu'v') 스칼라 회귀용 데이터셋.

        Parameters
        ----------
        use_lut_summary : bool
            True이면 LUT 요약(채널별 mean, 9pt에서의 LUT값)을 포함

        Returns
        -------
        X_mat : np.ndarray, shape (N * 12, Dx)
            행 단위 = (pk, macbeth_patch)
        y_vec : np.ndarray, shape (N * 12,)
            Δu'v' 값
        """
        X_rows, y_vals = [], []
        # LUT 9포인트 인덱스 (4096 기준)
        lut_points = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4095]
        
        for s in self.samples:
            Xd = s["X"]
            Yd = s["Y"]
            meta = Xd["meta"]
            
            feats = []
            # --- (1) 메타 데이터 ---
            feats.extend(meta['panel_maker'].astype(np.float32).tolist())
            feats.append(float(meta['frame_rate']))
            feats.append(float(meta['model_year']))

            # --- (2) LUT summary ---
            if use_lut_summary:
                lut = Xd["lut"]
                # 간단 요약: 채널별 mean, 9 포인트 값
                for ch in ["R_Low", "R_High", "G_Low", "G_High", "B_Low", "B_High"]:
                    arr = np.asarray(lut[ch], dtype=np.float32)
                    arr = np.clip(arr, 0.0, 1.0)  # 안전 장치
                    
                    # mean
                    feats.append(float(arr.mean()))
                    # 9포인트 샘플 (256포인트 기준으로 리샘플되어 있음)
                    n = len(arr)
                    for p in lut_points:
                        # 4096기준 인덱스 → 256포인트 기준으로 보정
                        idx = int(round(p / 16))  # 4096→256 매핑
                        idx = min(max(idx, 0), n - 1)
                        feats.append(float(arr[idx]))

            feats = np.asarray(feats, dtype=np.float32)

            # --- (3) Macbeth 12패치 반복 ---
            for patch in _MACBETH_LIST:
                X_rows.append(feats)                    # 동일 메타/요약으로 12행 생성
                y_vals.append(float(Yd['Y2'][patch]))

        X_mat = np.vstack(X_rows).astype(np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        return X_mat, y_vec

    # -----------------------------------------------------
    # 5) 잔차 보정용: 1차 모델 예측 벡터로부터 y_true - y_pred 생성
    # -----------------------------------------------------
    def build_residual_dataset(self, builder_fn, base_pred, **builder_kwargs):
        """
        잔차 보정용 데이터셋 (예: 선형 회귀 → RF 잔차 보정)

        Parameters
        ----------
        builder_fn : callable
            예) self.build_per_gray_y0, self.build_per_gray_y1 등
        base_pred : array-like
            1차 모델 예측 벡터 (y_true와 동일한 순서/길이)
        builder_kwargs : dict
            builder_fn에 전달할 파라미터

        Returns
        -------
        X_mat : np.ndarray, shape (M, Dx)
        y_resid : np.ndarray, shape (M,)
            y_true - base_pred
        """
        X_mat, y_true = builder_fn(**builder_kwargs)
        base_pred = np.asarray(base_pred, dtype=np.float32).reshape(-1)
        if base_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f"base_pred length {base_pred.shape[0]} != y_true length {y_true.shape[0]}")
        y_resid = (y_true - base_pred).astype(np.float32)
        return X_mat, y_resid

    # -----------------------------------------------------
    # (선택) 파이토치 호환: 길이/인덱싱
    # -----------------------------------------------------
    def __len__(self):
        # 파이토치 DataLoader로 직접 쓸 계획이 있다면,
        # 원하는 빌더 출력으로 커스텀 Dataset을 따로 구성하는 것을 권장합니다.
        return len(self.samples)

    def __getitem__(self, idx):
        # 원본 dict를 그대로 반환 (torch 학습 시엔 별도 빌더로 만든 X,y를 쓰세요)
        return self.samples[idx]
    
# if __name__ == "__main__":
#     import numpy as np
#     import pandas as pd
#     import tempfile, webbrowser
#     import matplotlib.pyplot as plt

#     target_pk = 500
#     reference_pk = 203
#     dataset = VACDataset([target_pk], reference_pk)

#     # 1) 추출
#     Y = dataset.samples[0]["Y"]
#     assert "Y0" in Y, "Y0_abs가 준비되어 있지 않습니다. prepare_output.py가 절대타깃(Gamma/Cx/Cy)으로 수정되었는지 확인하세요."
#     y0_abs = Y["Y0"]  # {'W':{'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)}, ...}

#     # 2) 긴형(long) 테이블로 펼치기 -> CSV로 열어보기 좋음
#     rows = []
#     for ptn, comps in y0_abs.items():  # ptn in ['W','R','G','B']
#         for comp_name in ['Gamma','Cx','Cy']:
#             arr = comps[comp_name]
#             for g, val in enumerate(arr):
#                 rows.append({"Pattern": ptn, "Component": comp_name, "Gray": g, "Value": val})
#     Y0abs_df = pd.DataFrame(rows)

#     # 3) 빠른 품질 체크: NaN 개수, 유효 구간 요약
#     summary = (
#         Y0abs_df
#         .groupby(["Pattern","Component"])
#         .agg(
#             count=("Value","size"),
#             nan_cnt=("Value", lambda s: int(np.isnan(s).sum())),
#             min_val=("Value", "min"),
#             max_val=("Value", "max"),
#             mean_val=("Value", "mean")
#         )
#         .reset_index()
#     )
#     print("\n[Y0_abs 요약]")
#     print(summary)

#     # 4) 경계 NaN 확인 (Gamma는 Gray 0/255에서 NaN이어야 함)
#     gamma_W = Y0abs_df[(Y0abs_df.Pattern=="W") & (Y0abs_df.Component=="Gamma")]
#     print("\n[체크] W 패턴 Gamma @Gray0, @Gray255:", gamma_W.loc[gamma_W.Gray.isin([0,255]), ["Gray","Value"]].to_dict("records"))

#     # 5) CSV로 저장 후 자동 열기(엑셀/기본 뷰어)
#     with tempfile.NamedTemporaryFile(delete=False, suffix="_Y0_abs.csv", mode="w", newline="", encoding="utf-8") as tmp:
#         Y0abs_df.to_csv(tmp.name, index=False)
#         print(f"\n[Y0_abs CSV] {tmp.name}")
#         webbrowser.open(f"file://{tmp.name}")

#     # 6) 빠른 시각화: 각 패턴별 Gamma 곡선(유효구간만)
#     try:
#         fig, axes = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
#         axes = axes.ravel()
#         for i, ptn in enumerate(['W','R','G','B']):
#             sub = Y0abs_df[(Y0abs_df.Pattern==ptn) & (Y0abs_df.Component=="Gamma")].sort_values("Gray")
#             x = sub["Gray"].to_numpy()
#             y = sub["Value"].to_numpy()
#             # 유효 구간만 그리기 (NaN 제거)
#             mask = ~np.isnan(y)
#             axes[i].plot(x[mask], y[mask])
#             axes[i].set_title(f"Gamma (abs) - {ptn}")
#             axes[i].set_xlabel("Gray")
#             axes[i].set_ylabel("Gamma")
#             axes[i].grid(True, alpha=0.3)
#         plt.show()
#     except Exception as e:
#         print(f"[Plot skipped] {e}")
    
# if __name__ == "__main__":
#     target_pk = 500
#     reference_pk = 203
#     dataset = VACDataset([target_pk], reference_pk)
    
    # X = dataset.samples[0]["X"]

    # X_df = pd.DataFrame({
    #     "R_Low":  X["lut"]["R_Low"],
    #     "R_High": X["lut"]["R_High"],
    #     "G_Low":  X["lut"]["G_Low"],
    #     "G_High": X["lut"]["G_High"],
    #     "B_Low":  X["lut"]["B_Low"],
    #     "B_High": X["lut"]["B_High"],
    # })

    # X_df["panel_maker"] = str(X["meta"]["panel_maker"])
    # X_df["frame_rate"] = X["meta"]["frame_rate"]
    # X_df["model_year"] = X["meta"]["model_year"]

    # with tempfile.NamedTemporaryFile(delete=False, suffix="X_raw.csv", mode="w", encoding="utf-8", newline="") as tmp:
    #     X_df.to_csv(tmp.name, index=False)
    #     webbrowser.open(f"file://{tmp.name}")
    
    # Y = dataset.samples[0]["Y"]
    # np.set_printoptions(threshold=20)
    # # === Y0 저장 ===
    # rows = []
    # for ptn, comps in Y["Y0"].items():
    #     for comp, arr in comps.items():
    #         for g, val in enumerate(arr):
    #             rows.append({"Pattern": ptn, "Component": comp, "Gray": g, "Value": val})
    # Y0_df = pd.DataFrame(rows)

    # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp:
    #     Y0_df.to_csv(tmp.name, index=False)
    #     webbrowser.open(f"file://{tmp.name}")

    # # === Y1 저장 ===
    # rows = []
    # for ptn, arr in Y["Y1"].items():
    #     for g, val in enumerate(arr):
    #         rows.append({"Pattern": ptn, "GraySeg": g, "Slope": val})
    # Y1_df = pd.DataFrame(rows)

    # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp:
    #     Y1_df.to_csv(tmp.name, index=False)
    #     webbrowser.open(f"file://{tmp.name}")

    # # === Y2 저장 ===
    # Y2_df = pd.DataFrame(list(Y["Y2"].items()), columns=["Patch", "Delta_uv"])
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp:
    #     Y2_df.to_csv(tmp.name, index=False)
    #     webbrowser.open(f"file://{tmp.name}")
