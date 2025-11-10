# VAC_dataset.py
import sys
import os
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from torch.utils.data import Dataset
from src.data_preparation.prepare_input import VACInputBuilder
from src.data_preparation.prepare_output import VACOutputBuilder
from config.db_config import engine

logging.basicConfig(level=logging.DEBUG)

MEASUREMENT_INFO_TABLE = "W_VAC_SET_Info"

class VACDataset(Dataset):
    def __init__(self, pk_list, ref_pk=2582, drop_use_flag_N: bool = True):

        self.pk_list_all = list(pk_list)
        self.ref_pk = int(ref_pk)
        self.feature_channels = ('R_High','G_High','B_High')
        
        if drop_use_flag_N:
            self.pk_list = self._filter_by_use_flag(self.pk_list_all)
        else:
            self.pk_list = list(pk_list)
            
        if not self.pk_list:
            logging.warning("[VACDataset] 유효한 pk_list가 비어 있습니다.")
            
        self.samples = []
        self._collect()
        
    def _filter_by_use_flag(pk_list):
        if not pk_list:
            return []
        
        pk_str = ",".join(str(int(pk)) for pk in pk_list)
        
        query = f"""
        SELECT `PK`, `Use_Flag`
        FROM `{MEASUREMENT_INFO_TABLE}`
        WHERE `PK` IN ({pk_str})
        """
        df = pd.read.sql(query, engine)
        
        if df.empty:
            logging.warning("[VACDataset] Use_Flag 조회 결과가 비었습니다. 입력 pk_list 전체를 사용합니다.")
            return pk_list
        
        valid = df[df["Use_Flag"] != "N"]["PK"].astype(int).tolist()
        dropped = sorted(set(pk_list) - set(valid))
        
        if dropped:
            logging.info(f"[VACDataset] Use_Flag='N' 이라 제외된 PK: {dropped}")
        else:
            logging.info(f"[VACDataset] Use_Flag='N' 으로 제외된 PK가 없습니다.")
            
        return valid

    def _collect(self):
        for pk in self.pk_list:
            x_builder = VACInputBuilder(pk)
            # X = raw ΔLUT @ CSV 매핑 인덱스 (정규화 없음)
            X = x_builder.prepare_X_delta_lut_with_mapping(ref_pk=self.ref_pk)

            # 참조 PK를 동일하게 써서 ΔY0 계산
            y_builder = VACOutputBuilder(pk, ref_pk=self.ref_pk)
            Y = y_builder.prepare_Y(y1_patterns=('W',))  # Y0만 써도 되지만 구조 유지

            self.samples.append({"pk": pk, "X": X, "Y": Y})

    def _build_features_for_gray(self, X_dict, gray: int) -> np.ndarray:
        """
        한 행(feature) 구성:
        [ΔR_High, ΔG_High, ΔB_High, panel_maker(one-hot), frame_rate, model_year, gray_norm(=g/255), LUT_index_j(g)]

        - ΔR_High, ΔG_High, ΔB_High: LUT index 매핑 포인트 기준 (LUT 값)-(ref LUT 값). normalize 안함!
        - panel_maker(one-hot), frame_rate, model_year: meta에서 그대로 가져옵니다.
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

    def build_XYdataset_for_jacobian_g(self, component='dGamma'):
        """
        White 패턴만 선택, y = dGamma/dCx/dCy (target - ref).
        X는 raw ΔLUT(High 3채널) + 메타 + gray_norm(+ pattern onehot=White) + LUT index
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
                feat_row = self._build_features_for_gray(X_dict=Xd, gray=g)
                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

if __name__ == "__main__":
    # ds = VACDataset(pk_list=[3002], ref_pk=2744)
    # X_mat, y_vec, groups = ds.build_XYdataset_for_jacobian_g(component='dCx')

    # print("X_mat shape:", X_mat.shape)
    # print("y_vec shape:", y_vec.shape)

    # # range(n)에서 n 행까지만 출력
    # for i in range(100):
    #     print(f"\n--- row {i} ---")
    #     print("X:", X_mat[i])
    #     print("y:", y_vec[i])
    pk_list = list(range(2743, 3003))
    ds = VACDataset(pk_list=pk_list, ref_pk=2744)
    print(self.pk_list)
    
