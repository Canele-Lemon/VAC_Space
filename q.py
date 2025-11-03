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

_PATTERN_LIST = ['W', 'R', 'G', 'B']

def _onehot(idx: int, size: int) -> np.ndarray:
    """
    one-hot 벡터 변환 encoder
    
    """
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

class VACDataset(Dataset):
    def __init__(self, pk_list, ref_vac_info_pk=2582):
        self.pk_list = list(pk_list)
        self.ref_vac_info_pk = int(ref_vac_info_pk)
        self.samples = []
        self._collect()

    def _collect(self):
        for pk in self.pk_list:
            x_builder = VACInputBuilder(pk)
            # X = raw ΔLUT @ CSV 매핑 인덱스 (정규화 없음)
            X = x_builder.prepare_X_delta_raw_with_mapping(ref_vac_info_pk=self.ref_vac_info_pk)

            # 참조 PK를 동일하게 써서 ΔY0 계산
            y_builder = VACOutputBuilder(pk, reference_pk=self.ref_vac_info_pk)
            Y = y_builder.prepare_Y(y1_patterns=('W',))  # Y0만 써도 되지만 구조 유지

            self.samples.append({"pk": pk, "X": X, "Y": Y})

    def _build_features_for_gray(self, X_dict, gray: int) -> np.ndarray:
        """
        한 행(feature) 구성:
        [ ΔLUT_selected_channels(g), panel_onehot..., frame_rate, model_year,
        gray_norm(=g/255), LUT_index_j(g) ]

        - ΔLUT_selected_channels: self.feature_channels 순서대로, raw 12bit delta @ gray
        - panel_onehot: meta['panel_maker']
        - frame_rate, model_year: meta에서 그대로
        - gray_norm: 0..1
        - LUT_index_j: mapping_j[gray] (0..4095), raw 그대로
        """
        # 1) 소스 참조
        delta_lut = X_dict["lut_delta_raw"]   # dict: ch -> (256,) float32 (raw delta at mapped indices)
        meta      = X_dict["meta"]            # dict: panel_maker(one-hot), frame_rate, model_year
        j_map     = X_dict["mapping_j"]       # (256,) int32, gray -> LUT index(0..4095)

        # 2) 채널 부분: 지정된 feature_channels만 사용 (보통 High 3채널)
        row = []
        for ch in self.feature_channels:
            row.append(float(delta_lut[ch][gray]))   # raw delta (정규화 없음)

        # 3) 메타 부착
        row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # 4) gray 위치 정보
        row.append(gray / 255.0)                    # gray_norm

        # 5) LUT 물리 인덱스(매핑) 정보
        j_idx = int(j_map[gray])                    # 0..4095, raw
        row.append(float(j_idx))

        return np.asarray(row, dtype=np.float32)

    def build_white_y0_delta(self, component='dGamma',
                             feature_channels=('R_High','G_High','B_High')):
        """
        White 패턴만 선택, y = dGamma/dCx/dCy (target - ref).
        X는 raw ΔLUT(High 3채널) + 메타 + gray_norm(+ pattern onehot=White).
        """
        assert component in ('dGamma','dCx','dCy')

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
            Yd  = s["Y"]  # {"Y0": {...}, ...}

            p = 'W'
            y_vec = Yd['Y0'][p][component]  # (256,)
            for g in range(256):
                y_val = y_vec[g]
                if not np.isfinite(y_val):
                    continue
                feat_row = self._build_features_for_gray(
                    X_dict=Xd, gray=g, add_pattern=p, feature_channels=feature_channels
                )
                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

    # (선택) 파이토치 호환
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]



if __name__ == "__main__":
    ds = VACDataset(pk_list=[2635], ref_vac_info_pk=2582)
    X_mat, y_vec, groups = ds.build_white_y0_delta(component='dCx',
                                                   feature_channels=('R_High','G_High','B_High'))

    print("X_mat shape:", X_mat.shape)
    print("y_vec shape:", y_vec.shape)

    # 앞 몇 행만 출력
    for i in range(5):
        print(f"\n--- row {i} ---")
        print("X:", X_mat[i])
        print("y:", y_vec[i])

여기서 아래 에러가 떠요:
PS D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module> & C:/python310/python.exe "d:/00 업무/00 가상화기술/00 색시야각 보상 최적화/VAC algorithm/module/scripts/VAC_dataset.py"
Traceback (most recent call last):
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\VAC_dataset.py", line 124, in <module>
    X_mat, y_vec, groups = ds.build_white_y0_delta(component='dCx',
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\VAC_dataset.py", line 104, in build_white_y0_delta
    feat_row = self._build_features_for_gray(
TypeError: VACDataset._build_features_for_gray() got an unexpected keyword argument 'add_pattern'
