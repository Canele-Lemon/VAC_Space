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
            X = x_builder.prepare_X_delta()   # {"dLUT": {...}, "meta": {...}}
            Y = y_builder.prepare_Y(y1_patterns=('W',))    # {"Y0": {...}, "Y1": {...}, "Y2": {...}}
            
            self.samples.append({
                "pk": pk, 
                "X": X, 
                "Y": Y
            })

    # -----------------------------------------------------
    # 내부: 특정 gray에서 피처 벡터 만들기
    # -----------------------------------------------------
    def _build_features_for_gray(self, X_dict, gray: int, add_pattern: str | None = None,
                                 channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High')) -> np.ndarray:
        """
        특정 gray(0~255)에서 [6채널 LUT 값 + 메타 + gray_norm + (옵션)패턴 원핫]을 붙여 피처 구성.

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
    def build_per_gray_y0_delta(self, component='dGamma', patterns=('W','R','G','B')):
        """
        자코비안/보정 학습용 1D 회귀 데이터셋 생성.

        각 row는 (pk, pattern p, gray g)에 해당.
        X_row 는 ΔLUT 기반 피처 (prepare_X_delta() 결과에서 나온 lut)
        y_val 는 Δ응답 (dGamma / dCx / dCy), 즉 target - ref

        Parameters
        ----------
        component : {'dGamma','dCx','dCy'}
        patterns  : tuple of patterns to include ('W','R','G','B')

        Returns
        -------
        X_mat : (N, D)
        y_vec : (N,)
        groups: (N,) pk ID for each row (useful for grouped CV 등)
        """
        assert component in ('dGamma','dCx','dCy')

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # this is now ΔLUT dict (prepare_X_delta)
            Yd  = s["Y"]  # this is now ΔY dict (prepare_Y -> compute_Y0_struct)

            for p in patterns:
                y_vec = Yd['Y0'][p][component]  # (256,)
                for g in range(256):
                    y_val = y_vec[g]
                    if not np.isfinite(y_val):
                        continue

                    feat_row = self._build_features_for_gray(
                        X_dict=Xd,
                        gray=g,
                        add_pattern=p
                    )

                    X_rows.append(feat_row)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        if X_rows:
            X_mat = np.vstack(X_rows).astype(np.float32)
        else:
            X_mat = np.empty((0,0), dtype=np.float32)

        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)

        return X_mat, y_vec, groups

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
    
def debug_dump_delta_training_rows():
    # 1) PK=2444만으로 dataset 구성
    ds = VACDataset(pk_list=[2444])

    # 2) ΔGamma 학습셋 생성
    X_mat, y_vec, groups = ds.build_per_gray_y0_delta(component='dGamma', patterns=('W',))

    print("[DEBUG] dGamma dataset (ΔLUT -> ΔGamma)")
    print("X_mat shape:", X_mat.shape)   # 예상: (유효 gray 수, feature_dim)
    print("y_vec shape:", y_vec.shape)   # 예상: (유효 gray 수,)
    print("groups shape:", groups.shape) # 모든 값이 2444일 것

    if X_mat.shape[0] == 0:
        print("No valid samples (all NaN?). Check measurement data or gamma calc.")
        return

    # panel one-hot 길이 파악
    panel_len = len(ds.samples[0]["X"]["meta"]["panel_maker"])

    # 3) 앞에서 몇 개만 출력
    for i in range(min(5, X_mat.shape[0])):
        print(f"\n--- sample {i} ---")
        print("pk:", groups[i])
        print("y (ΔGamma vs ref):", y_vec[i])

        feat = X_mat[i]

        # feat layout:
        # [ΔR_Low, ΔR_High, ΔG_Low, ΔG_High, ΔB_Low, ΔB_High,
        #  panel_onehot..., frame_rate, model_year,
        #  gray_norm,
        #  pattern_onehot(4)]

        delta_lut_part = feat[:6]
        panel_oh       = feat[6 : 6+panel_len]
        frame_rate     = feat[6+panel_len]
        model_year     = feat[6+panel_len+1]
        gray_norm      = feat[6+panel_len+2]
        pattern_onehot = feat[6+panel_len+3 : 6+panel_len+7]

        print("ΔLUT[0:6]             :", delta_lut_part)
        print("panel_onehot          :", panel_oh)
        print("frame_rate            :", frame_rate)
        print("model_year            :", model_year)
        print("gray_norm             :", gray_norm)
        print("pattern_onehot(WRGB)  :", pattern_onehot)

    print("\n[CHECK]")
    print("- ΔLUT[0:6]는 prepare_X_delta()에서 본 delta lut 값과 동일해야 합니다 (같은 gray 인덱스).")
    print("- y는 ΔGamma = target_gamma - ref_gamma 이므로 0에 가까우면 레퍼런스와 유사.")
    print("- gray_norm가 0.5 근처면 gray≈128 정도 샘플일 거고, pattern_onehot이 [1,0,0,0]이면 'W' 패턴입니다.")

def debug_preview_training_rows(pk_list=[2444], component='dGamma', patterns=('W',), max_print=5):
    """
    pk_list에 있는 panel들로 VACDataset을 만들고
    build_per_gray_y0() 결과 중 앞 max_print개 행을 사람이 읽을 수 있게 출력.

    component: 'dGamma' | 'dCx' | 'dCy'
    patterns : ('W',) 등으로 제한 가능
    """
    ds = VACDataset(pk_list=pk_list)

    X_mat, y_vec, groups = ds.build_per_gray_y0_delta(component=component, patterns=patterns)

    print("[DEBUG] build_per_gray_y0 result")
    print("  X_mat shape:", X_mat.shape)
    print("  y_vec shape:", y_vec.shape)
    print("  groups shape:", groups.shape)

    # panel onehot 길이
    panel_len = len(ds.samples[0]["X"]["meta"]["panel_maker"])

    for i in range(min(max_print, X_mat.shape[0])):
        feat = X_mat[i]
        y    = y_vec[i]
        pk   = groups[i]

        # feat layout 복원
        # 0:6 = ΔR_Low, ΔR_High, ΔG_Low, ΔG_High, ΔB_Low, ΔB_High
        delta_lut_6 = feat[:6]

        # panel onehot 다음 위치들 계산
        idx_panel_start = 6
        idx_panel_end   = 6 + panel_len
        panel_oh        = feat[idx_panel_start:idx_panel_end]

        frame_rate      = feat[idx_panel_end + 0]
        model_year      = feat[idx_panel_end + 1]
        gray_norm       = feat[idx_panel_end + 2]

        pattern_onehot  = feat[idx_panel_end + 3 : idx_panel_end + 7]

        # gray 추정도 해볼 수 있음: gray_norm ~= g/255
        est_gray = gray_norm * 255.0

        # pattern 추정: argmax of pattern_onehot
        ptn_idx = int(np.argmax(pattern_onehot))
        ptn_map = ['W','R','G','B']
        ptn     = ptn_map[ptn_idx]

        print(f"\n--- sample {i} ---")
        print(f"pk: {pk}")
        print(f"pattern_onehot -> {pattern_onehot}  => pattern '{ptn}'")
        print(f"gray_norm      -> {gray_norm:.6f}  (≈ gray {est_gray:.1f})")
        print(f"ΔLUT[0:6]      -> [ΔR_Low, ΔR_High, ΔG_Low, ΔG_High, ΔB_Low, ΔB_High]")
        print(f"                  {delta_lut_6}")
        print(f"panel_onehot   -> {panel_oh}")
        print(f"frame_rate     -> {frame_rate}")
        print(f"model_year     -> {model_year}")
        print(f"target {component} (y) -> {y}")
        
if __name__ == "__main__":
    debug_preview_training_rows()
