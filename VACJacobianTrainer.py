# VACJacobianTrainer.py
import os, sys, joblib, json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 프로젝트 경로
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.VAC_dataset import VACDataset  # 이미 사용 중인 Dataset

# ------------------------
# 설정
# ------------------------
KNOTS = 33                     # High/Low 컨트롤 포인트 개수(권장: 9~33)
PATTERNS = ['W']   # Y0의 패턴
COMPONENTS = ['Gamma','Cx','Cy']
RANDOM_STATE = 42

save_dir = os.path.dirname(__file__)
out_path = os.path.join(save_dir, "jacobian_Y0_high_K33.pkl")

# ------------------------
# 유틸: knot & basis
# ------------------------
# jacobian_train_offline.py (혹은 당신의 학습 스크립트 상단)
def build_pk_list_for_jacobian():
    blocks = [
        range(1411, 1934),   # 1411~1933
        range(1935, 2154),    # 1935~2153
        range(2155, 2455),     # 2155~2454
    ]
    pk_set = set()
    for b in blocks:
        pk_set.update(b)
    pk_list = sorted(pk_set)
    print(f"[Jacobian] using {len(pk_list)} PKs")
    return pk_list

def make_knot_positions(L=256, K=KNOTS) -> np.ndarray:
    # [0..255] 구간에서 동일 간격 노드 인덱스
    return np.round(np.linspace(0, L-1, K)).astype(int)

def linear_interp_weights(g: int, knots: np.ndarray) -> np.ndarray:
    """
    그레이 g(0..255)에 대해, K개 knot에 대한 선형보간 '모자(hat)' 가중치 벡터 φ(g) 반환.
    - 양 끝은 1개, 중간은 2개 노드만 비영(희소)
    """
    K = len(knots)
    w = np.zeros(K, dtype=np.float32)
    # 왼쪽/오른쪽 경계
    if g <= knots[0]:
        w[0] = 1.0
        return w
    if g >= knots[-1]:
        w[-1] = 1.0
        return w
    # 내부: 인접한 두 knot 사이
    i = np.searchsorted(knots, g) - 1
    g0, g1 = knots[i], knots[i+1]
    t = (g - g0) / max(1, (g1 - g0))
    w[i]   = 1.0 - t
    w[i+1] = t
    return w

def stack_basis_all_grays(knots: np.ndarray, L=256) -> np.ndarray:
    """
    모든 그레이(0..255)에 대한 φ(g) K차원 가중치 행렬 (L x K)
    """
    rows = [linear_interp_weights(g, knots) for g in range(L)]
    return np.vstack(rows).astype(np.float32)

# ------------------------
# 피처 구성
# ------------------------
@dataclass
class FeatureSlices:
    high_R: slice
    high_G: slice
    high_B: slice
    low_R:  slice
    low_G:  slice
    low_B:  slice
    meta:   slice          # panel_onehot + frame_rate + model_year
    gray:   int            # meta slice 내부에서 gray_norm 위치 (필요시 참조)
    pattern_oh: slice      # 4-dim

def build_feature_vector_for_gray(X_dict: dict, pattern: str, g: int,
                                  knots: np.ndarray, include_controls=True) -> Tuple[np.ndarray, FeatureSlices]:
    """
    입력:
      - X_dict: VACInputBuilder.prepare_X0() 결과
      - pattern: 'W','R','G','B' (패턴 원핫 포함)
      - g: 0..255
      - knots: K개 노드 인덱스
    출력:
      - feat: [High(R/G/B) knot-basis, (옵션) Low(R/G/B) knot-basis, meta, gray_norm, pattern_onehot(4)]
      - slices: 각 구간의 위치정보
    """
    lut = X_dict['lut']; meta = X_dict['meta']
    K = len(knots)

    # 1) High/Low knot 값 추출 (현재 곡선에서 노드 위치 샘플)
    #    - 학습 시엔 '현 상태 절대값'을 쓰지만, 선형모델이므로 계수는 '변화'에 선형입니다.
    H_R = lut['R_High'][knots]; H_G = lut['G_High'][knots]; H_B = lut['B_High'][knots]
    L_R = lut['R_Low'][knots];  L_G = lut['G_Low'][knots];  L_B = lut['B_Low'][knots]

    # 2) g에서의 보간 가중치 φ(g)
    phi = linear_interp_weights(g, knots)  # (K,)

    # 3) High/Low 기저(특징): 채널별로 φ(g) ⊙ knot-value
    #    - 자코비안 관점에선 'knot value' 자체가 독립변수이므로, 특징으로 φ(g)만 써도 됨.
    #      여기서는 "현재 knot값"도 함께 쓰는 절충안(현실 데이터 적합력↑). 
    #    - 순전히 자코비안만 원하면, 아래에서 H_R/G/B 대신 'phi 자체'를 쓰고
    #      나중에 계수 해석 시 β가 자코비안 그 자체가 됩니다(권장: 'phi-only' 모드).
    # ---- [권장] phi-only 모드 ----
    use_phi_only = True

    feats = []
    s = {}

    # High(R/G/B)
    idx0 = len(feats)
    if use_phi_only:
        feats.extend(phi); feats.extend(phi); feats.extend(phi)  # R,G,B
    else:
        feats.extend(phi * H_R); feats.extend(phi * H_G); feats.extend(phi * H_B)
    s_high_R = slice(idx0, idx0+K); idx0 += K
    s_high_G = slice(idx0, idx0+K); idx0 += K
    s_high_B = slice(idx0, idx0+K); idx0 += K

    # Low(R/G/B) - 컨트롤 피처
    if include_controls:
        if use_phi_only:
            feats.extend(phi); feats.extend(phi); feats.extend(phi)
        else:
            feats.extend(phi * L_R); feats.extend(phi * L_G); feats.extend(phi * L_B)
        s_low_R = slice(idx0, idx0+K); idx0 += K
        s_low_G = slice(idx0, idx0+K); idx0 += K
        s_low_B = slice(idx0, idx0+K); idx0 += K
    else:
        s_low_R = s_low_G = s_low_B = slice(idx0, idx0)

    # meta: panel_onehot + frame_rate + model_year
    panel = np.asarray(meta['panel_maker'], dtype=np.float32)
    feats.extend(panel.tolist())
    feats.append(float(meta['frame_rate']))
    feats.append(float(meta['model_year']))
    s_meta = slice(idx0, idx0 + len(panel) + 2); idx0 = s_meta.stop

    # gray_norm
    gray_norm = g/255.0
    feats.append(gray_norm)
    gray_pos = idx0; idx0 += 1

    # pattern one-hot (4)
    p_oh = np.zeros(4, np.float32)
    p_oh[['W','R','G','B'].index(pattern)] = 1.0
    feats.extend(p_oh.tolist())
    s_poh = slice(idx0, idx0+4); idx0 += 4

    feat = np.asarray(feats, dtype=np.float32)
    slices = FeatureSlices(
        high_R=s_high_R, high_G=s_high_G, high_B=s_high_B,
        low_R=s_low_R,   low_G=s_low_G,   low_B=s_low_B,
        meta=s_meta, gray=gray_pos, pattern_oh=s_poh
    )
    return feat, slices

# ------------------------
# 데이터셋 구축 (Y0 절대)
# ------------------------
def build_dataset_Y0_abs(ds: VACDataset, knots: np.ndarray,
                         components=('Gamma','Cx','Cy')) -> Tuple[np.ndarray, np.ndarray, FeatureSlices]:
    """
    행 단위: (pk, pattern, gray)
    X: [High φ(g) R/G/B | Low φ(g) R/G/B | meta | gray_norm | pattern_oh]
    y: 선택한 컴포넌트(감마/Cx/Cy) 스칼라
    - Gamma의 gray=0/255 및 NaN은 드롭
    """
    X_rows, y_vals = [], []
    slices_keep = None

    for s in ds.samples:
        Xd, Yd = s['X'], s['Y']
        for p in PATTERNS:
            for comp in components:
                arr = Yd['Y0'][p][comp]  # (256,)
                for g in range(256):
                    val = arr[g]
                    if comp == 'Gamma' and (g < 1 or g > 254):
                        continue
                    if not np.isfinite(val):   # NaN, inf 제거 (Gamma@0/255 등)
                        continue
                    feat, sli = build_feature_vector_for_gray(Xd, p, g, knots, include_controls=True)
                    X_rows.append(feat)
                    y_vals.append(float(val))
                    if slices_keep is None:
                        slices_keep = sli

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_vals, dtype=np.float32)
    return X, y, slices_keep

# ------------------------
# 학습 / 저장
# ------------------------
import time

def train_jacobian_models(pk_list: List[int], save_path: str, knots_K=KNOTS):
    start_total = time.time()

    knots = make_knot_positions(K=knots_K)
    ds = VACDataset(pk_list)

    print(f"\n[Jacobian] Start training with {len(pk_list)} PKs, {knots_K} knots")

    artifacts = {"knots": knots.tolist(), "components": {}}

    for comp in COMPONENTS:
        print(f"\n=== Learn Jacobian for Y0-{comp} (vs High) ===")
        t0 = time.time()

        # 데이터 구축
        X, y, feat_slices = build_dataset_Y0_abs(ds, knots, components=(comp,))
        print(f"  └ X shape: {X.shape}, y shape: {y.shape}")

        # 학습 (표준화 + Ridge)
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge",  Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ])
        model.fit(X, y)

        # 시간 측정
        t1 = time.time()
        print(f"  ⏱  Y0-{comp} done in {(t1 - t0):.1f} s")

        ridge = model.named_steps["ridge"]
        scaler = model.named_steps["scaler"]
        coef = ridge.coef_.astype(np.float32)

        artifacts["components"][comp] = {
            "coef": coef.tolist(),
            "intercept": float(ridge.intercept_),
            "feature_slices": {
                "high_R": [feat_slices.high_R.start, feat_slices.high_R.stop],
                "high_G": [feat_slices.high_G.start, feat_slices.high_G.stop],
                "high_B": [feat_slices.high_B.start, feat_slices.high_B.stop],
                "low_R":  [feat_slices.low_R.start,  feat_slices.low_R.stop],
                "low_G":  [feat_slices.low_G.start,  feat_slices.low_G.stop],
                "low_B":  [feat_slices.low_B.start,  feat_slices.low_B.stop],
                "meta":   [feat_slices.meta.start,   feat_slices.meta.stop],
                "gray":   feat_slices.gray,
                "pattern_oh": [feat_slices.pattern_oh.start, feat_slices.pattern_oh.stop],
            },
            "standardizer": {
                "mean": scaler.mean_.astype(np.float32).tolist(),
                "scale": scaler.scale_.astype(np.float32).tolist()
            }
        }

    total_time = time.time() - start_total
    print(f"\n✅ All components trained in {total_time/60:.1f} min")

    joblib.dump(artifacts, save_path, compress=("gzip", 3))
    print(f"📁 saved Jacobian model: {save_path}")

# ------------------------
# 자코비안 A 생성 함수
# ------------------------
def build_A_for_component(artifacts: dict, comp: str, L=256) -> np.ndarray:
    """
    ΔY ≈ A Δh  (여기서 Δh는 [R_high_knots(K), G_high_knots(K), B_high_knots(K)] 순서)
    - 학습 시 'phi-only' 모드였으므로: A[g, k] = β_k * φ_k(g)
    - R/G/B를 좌우로 이어붙여 A의 열 수 = 3K
    - 패턴/메타를 고정하지 않고 '기저 기반'의 평균적 자코비안(모형 계수만) 반환
      (운용 시 패턴별 A가 필요하면, pattern 원핫을 고정하고 “해당 패턴 행만 사용”)
    """
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    comp_obj = artifacts["components"][comp]
    coef = np.asarray(comp_obj["coef"], dtype=np.float32)
    scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)

    s = comp_obj["feature_slices"]
    s_high_R = slice(s["high_R"][0], s["high_R"][1])
    s_high_G = slice(s["high_G"][0], s["high_G"][1])
    s_high_B = slice(s["high_B"][0], s["high_B"][1])

    beta_R = coef[s_high_R] / np.maximum(scale[s_high_R], 1e-12)
    beta_G = coef[s_high_G] / np.maximum(scale[s_high_G], 1e-12)
    beta_B = coef[s_high_B] / np.maximum(scale[s_high_B], 1e-12)

    Phi = stack_basis_all_grays(knots, L=L)  # (L,K)

    # A = [Phi * diag(beta_R) | Phi * diag(beta_G) | Phi * diag(beta_B)]
    A_R = Phi * beta_R.reshape(1, -1)
    A_G = Phi * beta_G.reshape(1, -1)
    A_B = Phi * beta_B.reshape(1, -1)
    A = np.hstack([A_R, A_G, A_B]).astype(np.float32)  # (L, 3K)
    return A
    
def main():
    # 1) PK 목록 구성
    # full_pk_range = list(range(81, 909))
    # exclude = [203, 604, 605, 853, 855, 856]
    # pk_list = [pk for pk in full_pk_range if pk not in exclude]
    pk_list = build_pk_list_for_jacobian()

    # 3) 자코비안 학습 실행 (예: Y0=Gamma/Cx/Cy 3개 모두 High만의 영향)
    #    함수명은 당신이 사용 중인 오프라인 학습 함수로 바꾸세요.
    train_jacobian_models(pk_list, out_path, knots_K=KNOTS)

if __name__ == "__main__":
    main()
