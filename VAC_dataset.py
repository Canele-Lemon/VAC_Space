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
        
        if drop_use_flag_N:
            self.pk_list = self._filter_by_use_flag(self.pk_list_all)
        else:
            self.pk_list = list(pk_list)
            
        if not self.pk_list:
            logging.warning("[VACDataset] 유효한 pk_list가 비어 있습니다.")
            
        self.samples = []
        self._collect()
        
    def _filter_by_use_flag(self, pk_list):
        if not pk_list:
            return []
        
        pk_str = ",".join(str(int(pk)) for pk in pk_list)
        
        query = f"""
        SELECT `PK`, `Use_Flag`
        FROM `{MEASUREMENT_INFO_TABLE}`
        WHERE `PK` IN ({pk_str})
        """
        df = pd.read_sql(query, engine)
        
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

    def _build_features_for_gray(self, X_dict, gray: int, channels) -> np.ndarray:
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
        for ch in channels:
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

    def _build_XY0_for_jacobian_g(self, component='dGamma'):
        """
        - X: raw ΔLUT(High 3채널) + meta + gray_norm + LUT index
        - y: dGamma / dCx / dCy (White 패턴, target - ref)
        """
        assert component in ('dGamma','dCx','dCy')

        jac_channels = ('R_High', 'G_High', 'B_High')
        
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
                    X_dict=Xd, 
                    gray=g,
                    channels=jac_channels
                )
                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups
    
    def _build_XY0(
        self,
        component: str = "dGamma",
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W',),
    ):
        """
        Y0 예측용(X→ΔGamma/ΔCx/ΔCy) per-gray 데이터셋.
        - X: ΔLUT(지정 채널) + meta + gray_norm + LUT index
        - y: 선택한 component (ΔGamma/ΔCx/ΔCy), 지정된 패턴들(W/R/G/B)
        """
        assert component in ('dGamma', 'dCx', 'dCy')

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
            Yd  = s["Y"]  # {"Y0": {...}, "Y1": {...}, "Y2": {...}}

            for p in patterns:
                if "Y0" not in Yd or p not in Yd["Y0"]:
                    continue
                y_vec = Yd["Y0"][p][component]  # (256,)
                for g in range(256):
                    y_val = y_vec[g]
                    if not np.isfinite(y_val):
                        continue
                    feat_row = self._build_features_for_gray(
                        X_dict=Xd,
                        gray=g,
                        channels=channels,
                    )
                    X_rows.append(feat_row)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups
    
    def build_XY_dataset(
        self,
        target: str,
        component: str | None = None,
        channels=None,
        patterns=('W',),
    ):
        """
        통합 XY 데이터셋 빌더.

        Parameters
        ----------
        target : {'Y0', 'Y1', 'Y2', 'jacobian'}
            어떤 타겟을 예측할지 선택.
        component : str | None
            - target='Y0' 또는 'jacobian' 일 때: {'dGamma','dCx','dCy'}
            - target='Y1','Y2' 에서는 사용 안 함(또는 향후 확장 용도).
        channels : tuple[str] | None
            X에 사용할 LUT 채널 리스트.
            - target='jacobian' 인 경우: None이면 ('R_High','G_High','B_High')
            - target='Y0' 인 경우: None이면 ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
        patterns : tuple[str]
            사용할 패턴 (지금은 보통 ('W',) 로 쓰는 걸 가정)

        Returns
        -------
        X_mat : np.ndarray
        y_vec : np.ndarray
        groups : np.ndarray
        """
        target = target.lower()

        if target == "jacobian":
            # 자코비안: High 3채널 고정, 기존 메서드 재사용
            if component is None:
                component = "dGamma"
            return self._build_XY0_for_jacobian_g(component=component)

        if target == "y0":
            if component is None:
                raise ValueError("target='Y0'일 때 component('dGamma','dCx','dCy')가 필요합니다.")
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY0(component=component, channels=channels, patterns=patterns)

        if target == "y1":
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY1(channels=channels, patterns=patterns)

        if target == "y2":
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY2(channels=channels)

        raise ValueError(f"Unknown target='{target}'. (지원: 'jacobian','Y0','Y1','Y2')")

if __name__ == "__main__":
    pk_list = [3008]
    BYPASS_PK = 3007
    
    dataset = VACDataset(pk_list=pk_list, ref_pk=BYPASS_PK)
    
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
    
    X_dG, y_dG, grp_dG = dataset.build_XY_dataset(
        target="Y0",
        component="dGamma",
        channels=channels,
        patterns=('W',),
    )
    
    print("X_dG shape:", X_dG.shape)
    print("y_dG shape:", y_dG.shape)
    print("groups shape:", grp_dG.shape)
    
    # -------- 앞 몇 행만 'dataset 느낌'으로 보기 --------
    n_preview = min(30, X_dG.shape[0])  # 30행까지만

    # panel_maker one-hot 길이 가져오기 (첫 샘플 meta 이용)
    first_meta = dataset.samples[0]["X"]["meta"]
    panel_dim = len(first_meta["panel_maker"])

    feature_names = (
        [f"d{ch}" for ch in channels] +              # ΔLUT (Low/High 6채널)
        [f"panel_{i}" for i in range(panel_dim)] +   # panel maker one-hot
        ["frame_rate", "model_year", "gray_norm", "LUT_j"]
    )

    df = pd.DataFrame(X_dG[:n_preview, :], columns=feature_names)
    df["y"] = y_dG[:n_preview]
    df["pk_group"] = grp_dG[:n_preview]

    # 예쁘게 출력
    print("\n[PREVIEW] Y0-dGamma XY dataset (first rows)")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .6f}"))


네 그러면 dCx, dCy 데이터셋은 0~5 gray는 학습에서 제외할 수 있도록 위 코드를 수정해주시고 _build_XY1과 _build_XY2는 다음과 같은 조건으로 만들게요.
_build_XY1: per-segment이기 때문에 중앙 gray의 델타 RGB Low LUT와 RGB High LUT 값으로 X를 뽑을게요. (예: 88-96 gray slope면 (88+96)/2 gray의 RGB LUT High와 Low 값. 그러면 행 수가 아마 15행이 되겠군여)
_build_XY2: _build_XY1와 동일하게 X를 구성할게요
