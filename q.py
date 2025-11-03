# VAC_dataset.py (핵심 부분만 교체/정리)

from torch.utils.data import Dataset
from src.prepare_input import VACInputBuilder
from src.prepare_output import VACOutputBuilder
import numpy as np

_PATTERN_LIST = ['W', 'R', 'G', 'B']

def _onehot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

class VACDataset(Dataset):
    """
    - 생성자 단순화: pk_list, ref_vac_info_pk만 받음
    - X: prepare_X_delta_raw_with_mapping() 결과 사용
         => {"lut_delta_raw": dict(6ch, len=256), "meta": {...}, "mapping_j": (256,)}
       (LUT index/값 모두 정규화 없음, CSV 매핑 j에서의 raw 차이)
    - Y: VACOutputBuilder(reference_pk=ref_vac_info_pk)로 계산한 ΔY0(dGamma/dCx/dCy)
    """

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

    def _build_features_for_gray(self, X_dict, gray: int,
                                 add_pattern: str | None = None,
                                 feature_channels=('R_High','G_High','B_High')) -> np.ndarray:
        """
        feature = [ΔR_High, ΔG_High, ΔB_High, panel_onehot..., frame_rate, model_year, gray/255, (opt)pattern_onehot]
        - Δ* 값은 raw 12bit 차이 (정규화 X)
        """
        lut_delta_raw = X_dict["lut_delta_raw"]  # ← raw
        meta = X_dict["meta"]

        row = [float(lut_delta_raw[ch][gray]) for ch in feature_channels]

        # 메타
        row.extend(np.asarray(meta["panel_maker"], dtype=np.float32).tolist())
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # gray는 피처 스케일러블 메타로 유지(정규화) — LUT index/값 정규화와는 별개
        row.append(gray / 255.0)

        if add_pattern is not None:
            pat_idx = _PATTERN_LIST.index(add_pattern) if add_pattern in _PATTERN_LIST else -1
            row.extend(_onehot(pat_idx, len(_PATTERN_LIST)).tolist())

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


# -------- 디버그 유틸 --------
def debug_preview_white_y0(pk_list, ref_vac_info_pk=2582,
                           component='dGamma',
                           feature_channels=('R_High','G_High','B_High'),
                           max_print=10):
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_vac_info_pk)
    X_mat, y_vec, groups = ds.build_white_y0_delta(component=component,
                                                   feature_channels=feature_channels)

    print("[DEBUG] WHITE-only Y0(Δ) dataset")
    print("  ref_vac_info_pk:", ref_vac_info_pk)
    print("  component      :", component)
    print("  feature_ch     :", feature_channels)
    print("  X_mat shape    :", X_mat.shape)
    print("  y_vec shape    :", y_vec.shape)
    print("  groups shape   :", groups.shape)

    if X_mat.shape[0] == 0:
        print("  -> No valid samples.")
        return

    panel_len = len(ds.samples[0]["X"]["meta"]["panel_maker"])
    F = len(feature_channels)

    for i in range(min(max_print, X_mat.shape[0])):
        feat = X_mat[i]; y = y_vec[i]; pk = groups[i]
        delta_lut = feat[:F]
        idx = F
        panel_oh = feat[idx: idx+panel_len]; idx += panel_len
        frame_rate = feat[idx]; idx += 1
        model_year = feat[idx]; idx += 1
        gray_norm  = feat[idx]; idx += 1
        pattern_oh = feat[idx: idx+4]
        est_gray   = gray_norm * 255.0
        pat_idx    = int(np.argmax(pattern_oh))
        pat_name   = _PATTERN_LIST[pat_idx] if 0 <= pat_idx < 4 else 'NA'

        print(f"\n--- sample {i} ---")
        print(f"pk: {pk} / pattern: {pat_name} / gray≈{est_gray:.1f}")
        print(f"ΔLUT(raw, {feature_channels}): {delta_lut}")  # ← 정수(또는 정수에 가까운) 12bit 스텝 차이
        print(f"panel_onehot : {panel_oh}")
        print(f"frame_rate   : {frame_rate}")
        print(f"model_year   : {model_year}")
        print(f"y ({component}) : {y}")


if __name__ == "__main__":
    # 예: PK=2635 대상, ref=2582, White에서 dCx 학습행 미리보기
    debug_preview_white_y0(pk_list=[2635], ref_vac_info_pk=2582,
                           component='dCx',
                           feature_channels=('R_High','G_High','B_High'),
                           max_print=10)