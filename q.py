import os, sys, time, joblib
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------
# 경로 세팅: src.* 불러올 수 있게
# ---------------------------------
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../module/scripts or .../module/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))    # .../module
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.VAC_dataset import VACDataset  # 경로는 실제 위치에 맞게 조정하세요 (scripts. 가 아니라 src.라면 그대로)

# ------------------------
# 설정
# ------------------------
KNOTS = 33
PATTERNS = ['W','R','G','B']      # 학습에 사용할 패턴
COMPONENTS = ['dGamma','dCx','dCy']
RANDOM_STATE = 42

save_dir = CURRENT_DIR
out_path = os.path.join(save_dir, "jacobian_delta_highlow_K33.pkl")

# ------------------------
# pk list
# ------------------------
def build_pk_list_for_jacobian():
    blocks = [
        range(1411, 1934),   # 1411~1933
        range(1935, 2154),   # 1935~2153
        range(2155, 2455),   # 2155~2454
    ]
    pk_set = set()
    for b in blocks:
        pk_set.update(b)
    pk_list = sorted(pk_set)
    print(f"[Jacobian] using {len(pk_list)} PKs")
    return pk_list

# ------------------------
# basis utils
# ------------------------
def make_knot_positions(L=256, K=KNOTS) -> np.ndarray:
    return np.round(np.linspace(0, L-1, K)).astype(int)

def linear_interp_weights(g: int, knots: np.ndarray) -> np.ndarray:
    """
    gray g (0..255)에 대해 knot별 선형보간 basis φ(g) (K,)
    """
    K = len(knots)
    w = np.zeros(K, dtype=np.float32)

    # 양 끝 처리
    if g <= knots[0]:
        w[0] = 1.0
        return w
    if g >= knots[-1]:
        w[-1] = 1.0
        return w

    # 내부
    i = np.searchsorted(knots, g) - 1
    g0, g1 = knots[i], knots[i+1]
    t = (g - g0) / max(1, (g1 - g0))
    w[i]   = 1.0 - t
    w[i+1] = t
    return w

def stack_basis_all_grays(knots: np.ndarray, L=256) -> np.ndarray:
    """
    모든 gray(0..255)에 대한 basis φ(g)를 쌓은 행렬 (256,K)
    """
    rows = [linear_interp_weights(g, knots) for g in range(L)]
    return np.vstack(rows).astype(np.float32)

# ------------------------
# feature slice tracker
# ------------------------
@dataclass
class FeatureSlices:
    high_R: slice
    high_G: slice
    high_B: slice
    low_R:  slice
    low_G:  slice
    low_B:  slice
    meta:   slice
    gray:   int
    pattern_oh: slice

# ------------------------
# gray별 feature 벡터 만들기 (phi-only, High+Low 포함)
# ------------------------
def build_feature_vector_for_gray_delta(
    X_dict: dict,
    pattern: str,
    g: int,
    knots: np.ndarray,
    include_low=True,
) -> Tuple[np.ndarray, FeatureSlices]:
    """
    X_dict: VACDataset.samples[i]["X"]  (prepare_X_delta() 결과)
      X_dict["lut"][ch] = ΔLUT[ch][0..255]  (R_Low, R_High, ...)

    우리는 'phi-only' 기저를 사용:
      - 각 채널(R_high/G_high/B_high/R_low/G_low/B_low)에 대해 φ(g) (K차원)
      - 메타(panel onehot, frame_rate, model_year)
      - gray_norm
      - pattern onehot(4)

    이렇게 만든 feat(row)는 Ridge에 들어갑니다.
    """
    phi = linear_interp_weights(g, knots)  # (K,)

    feats = []
    idx0 = 0

    # High 채널들
    s_high_R = slice(idx0, idx0+len(phi)); feats.extend(phi); idx0 += len(phi)
    s_high_G = slice(idx0, idx0+len(phi)); feats.extend(phi); idx0 += len(phi)
    s_high_B = slice(idx0, idx0+len(phi)); feats.extend(phi); idx0 += len(phi)

    # Low 채널들 (옵션이지만 기본 True)
    if include_low:
        s_low_R = slice(idx0, idx0+len(phi)); feats.extend(phi); idx0 += len(phi)
        s_low_G = slice(idx0, idx0+len(phi)); feats.extend(phi); idx0 += len(phi)
        s_low_B = slice(idx0, idx0+len(phi)); feats.extend(phi); idx0 += len(phi)
    else:
        s_low_R = slice(idx0, idx0)
        s_low_G = slice(idx0, idx0)
        s_low_B = slice(idx0, idx0)

    # meta
    panel_vec = np.asarray(X_dict["meta"]["panel_maker"], dtype=np.float32)
    panel_list = panel_vec.tolist()
    feats.extend(panel_list)
    feats.append(float(X_dict["meta"]["frame_rate"]))
    feats.append(float(X_dict["meta"]["model_year"]))
    s_meta = slice(idx0, idx0 + len(panel_list) + 2)
    idx0 = s_meta.stop

    # gray_norm
    gray_norm_val = g/255.0
    feats.append(gray_norm_val)
    gray_pos = idx0
    idx0 += 1

    # pattern onehot
    p_oh = np.zeros(4, np.float32)
    p_oh[['W','R','G','B'].index(pattern)] = 1.0
    feats.extend(p_oh.tolist())
    s_poh = slice(idx0, idx0+4)
    idx0 += 4

    feat = np.asarray(feats, dtype=np.float32)

    slices = FeatureSlices(
        high_R=s_high_R, high_G=s_high_G, high_B=s_high_B,
        low_R=s_low_R,   low_G=s_low_G,   low_B=s_low_B,
        meta=s_meta, gray=gray_pos, pattern_oh=s_poh
    )
    return feat, slices

# ------------------------
# 전체 (X,y) 생성
# ------------------------
def build_dataset_delta(
    ds: VACDataset,
    knots: np.ndarray,
    component='dGamma',
    patterns=('W','R','G','B')
):
    """
    ds: VACDataset(pk_list=...)
    component: 'dGamma'|'dCx'|'dCy'
    patterns : 수집할 패턴들 ('W','R','G','B' 등)
    return:
        X_mat : (N,D)
        y_vec : (N,)
        feat_slices_keep : FeatureSlices (첫 row 기준)
    """
    X_rows = []
    y_vals = []
    feat_slices_keep = None

    for sample in ds.samples:
        Xd = sample["X"]   # ΔLUT + meta (prepare_X_delta 결과)
        Yd = sample["Y"]   # ΔGamma/ΔCx/ΔCy 등 (compute_Y0_struct 결과)

        for p in patterns:
            y_arr = Yd["Y0"][p][component]  # shape (256,)
            for g in range(256):
                y_val = y_arr[g]

                # Gamma는 gray=0,255 등에서 NaN 가능 → 그냥 스킵
                if not np.isfinite(y_val):
                    continue

                feat, slices = build_feature_vector_for_gray_delta(
                    X_dict=Xd,
                    pattern=p,
                    g=g,
                    knots=knots,
                    include_low=True
                )
                X_rows.append(feat)
                y_vals.append(float(y_val))

                if feat_slices_keep is None:
                    feat_slices_keep = slices

    if len(X_rows) == 0:
        X_mat = np.empty((0,0), dtype=np.float32)
    else:
        X_mat = np.vstack(X_rows).astype(np.float32)

    y_vec = np.asarray(y_vals, dtype=np.float32)

    return X_mat, y_vec, feat_slices_keep

# ------------------------
# 학습 / 저장
# ------------------------
def train_jacobian_models(pk_list: List[int], save_path: str, knots_K=KNOTS):
    """
    1) VACDataset(pk_list) 로드  (ΔLUT / ΔY)
    2) component별(dGamma, dCx, dCy) Ridge 학습
    3) coef, intercept, scaler 등 저장
    """
    start_total = time.time()

    knots = make_knot_positions(K=knots_K)
    ds = VACDataset(pk_list)

    print(f"\n[Jacobian-DELTA] Start training with {len(pk_list)} PKs, {knots_K} knots")

    artifacts = {
        "knots": knots.tolist(),
        "components": {}
    }

    for comp in COMPONENTS:
        print(f"\n=== Learn Jacobian for {comp} (ΔLUT High+Low) ===")
        t0 = time.time()

        # 데이터 구성
        X, y, feat_slices = build_dataset_delta(
            ds=ds,
            knots=knots,
            component=comp,
            patterns=PATTERNS
        )
        print(f"  └ X shape: {X.shape}, y shape: {y.shape}")

        # 표준화 + Ridge
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge",  Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ])
        model.fit(X, y)

        t1 = time.time()
        print(f"  ⏱  {comp} done in {(t1 - t0):.1f} s")

        ridge  = model.named_steps["ridge"]
        scaler = model.named_steps["scaler"]
        coef   = ridge.coef_.astype(np.float32)

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
                "mean":  scaler.mean_.astype(np.float32).tolist(),
                "scale": scaler.scale_.astype(np.float32).tolist()
            }
        }

    total_time = time.time() - start_total
    print(f"\n✅ All components trained in {total_time/60:.1f} min")

    joblib.dump(artifacts, save_path, compress=("gzip", 3))
    print(f"📁 saved Jacobian model: {save_path}")

# ------------------------
# 자코비안 A 행렬 빌더 (운용 시 사용)
# ------------------------
def build_A_for_component(artifacts: dict, comp: str, L=256) -> np.ndarray:
    """
    ΔY ≈ A Δh  (자코비안)
    Δh 순서:
        [R_high_knots(K), G_high_knots(K), B_high_knots(K),
         R_low_knots(K),  G_low_knots(K),  B_low_knots(K)]
    따라서 A shape = (256, 6K)

    artifacts: train_jacobian_models() 결과 로드한 dict
    comp: 'dGamma' | 'dCx' | 'dCy'
    """
    knots   = np.asarray(artifacts["knots"], dtype=np.int32)
    compobj = artifacts["components"][comp]

    coef  = np.asarray(compobj["coef"], dtype=np.float32)
    scale = np.asarray(compobj["standardizer"]["scale"], dtype=np.float32)
    s     = compobj["feature_slices"]

    # helper
    def _beta(slice_bounds):
        i0, i1 = slice_bounds
        sl = slice(i0, i1)
        return coef[sl] / np.maximum(scale[sl], 1e-12)

    beta_high_R = _beta(s["high_R"])
    beta_high_G = _beta(s["high_G"])
    beta_high_B = _beta(s["high_B"])
    beta_low_R  = _beta(s["low_R"])
    beta_low_G  = _beta(s["low_G"])
    beta_low_B  = _beta(s["low_B"])

    Phi = stack_basis_all_grays(knots, L=L)  # (L,K)

    A_high_R = Phi * beta_high_R.reshape(1, -1)
    A_high_G = Phi * beta_high_G.reshape(1, -1)
    A_high_B = Phi * beta_high_B.reshape(1, -1)
    A_low_R  = Phi * beta_low_R.reshape(1, -1)
    A_low_G  = Phi * beta_low_G.reshape(1, -1)
    A_low_B  = Phi * beta_low_B.reshape(1, -1)

    A = np.hstack([
        A_high_R, A_high_G, A_high_B,
        A_low_R,  A_low_G,  A_low_B
    ]).astype(np.float32)

    return A

# ------------------------
# main (여기만 호출하시면 됩니다)
# ------------------------
def main():
    # 1) PK 목록 구성
    pk_list = build_pk_list_for_jacobian()

    # 2) 자코비안 학습 실행: ΔLUT -> dGamma/dCx/dCy
    train_jacobian_models(pk_list, out_path, knots_K=KNOTS)

    # (선택) 학습된 artifacts로 자코비안 행렬 하나 뽑아보기 (dGamma 기준)
    artifacts = joblib.load(out_path)
    A_dGamma = build_A_for_component(artifacts, "dGamma")  # (256, 6*KNOTS)
    print("[DEBUG] A_dGamma shape:", A_dGamma.shape)
    print("        first row few elems:", A_dGamma[0, :10])

if __name__ == "__main__":
    main()