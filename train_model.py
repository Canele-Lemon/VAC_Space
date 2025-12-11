# train_model.py
import sys
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform
from joblib import parallel_backend

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.VAC_dataset import VACDataset

RANDOM_STATE = 42

# 공통: 하이브리드 회귀 (Linear + RF residual) 학습 함수
def train_hybrid_regressor(X_all, y_all, groups, tag="", normalize_y=True):
    """
    입력:
      - X_all, y_all: 전체 데이터 (float32 권장)
      - groups: PK 단위 그룹 레이블 (길이 = X_all 행수)
    수행:
      1) Group hold-out (8:2) 분할
      2) 1단계: StandardScaler → Ridge 학습
      3) 2단계: RandomForestRegressor로 residual 학습 (RandomizedSearchCV + GroupKFold)
    반환:
      - artifacts(dict): linear_model, rf_residual, rf_best_params, metrics
    """
    # 1) Group hold-out split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups))
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    groups_train = groups[train_idx]
    
    # --- Target standardization (train set stats only)
    if normalize_y:
        y_mean = float(np.nanmean(y_train))
        y_std  = float(np.nanstd(y_train) + 1e-8)  # div-by-zero 방지
        y_train_s = (y_train - y_mean) / y_std
    else:
        y_mean, y_std = 0.0, 1.0
        y_train_s = y_train

    # 2) Linear step: Standardize + Ridge
    linear_model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge",  Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])
    t0 = time.time()
    linear_model.fit(X_train, y_train_s)
    t1 = time.time()
    
    # 예측 (표준화 스페이스)
    base_pred_train_s = linear_model.predict(X_train)
    base_pred_test_s  = linear_model.predict(X_test)
    
    # 원-스케일로 복원
    base_pred_test = base_pred_test_s * y_std + y_mean
    base_mse = mean_squared_error(y_test, base_pred_test)
    base_r2  = r2_score(y_test, base_pred_test)
    print(f"⏱️ [{tag}] Linear fit: {(t1 - t0):.1f}s | MSE={base_mse:.6f} R²={base_r2:.6f}")

    # 3) RF residual step (표준화 스페이스에서 residual 학습)
    resid_train_s = (y_train_s - base_pred_train_s).astype(np.float32)
    
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,        # 내부 병렬 OFF (CV에서 바깥 병렬)
        bootstrap=True
    )
    param_dist = {
        "n_estimators":     randint(120, 300),
        "max_depth":        randint(8, 18),
        "min_samples_split":randint(2, 8),
        "min_samples_leaf": randint(4, 20),
        "max_features":     uniform(0.2, 0.8),
    }
    gkf = GroupKFold(n_splits=3)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=gkf,
        scoring="r2",
        n_jobs=-1,
        pre_dispatch="2*n_jobs",
        verbose=2,
        random_state=RANDOM_STATE,
        error_score=np.nan,
        return_train_score=True
    )
    t0 = time.time()
    with parallel_backend("threading", n_jobs=-1):
        search.fit(X_train, resid_train_s, groups=groups_train)
    t_resid = time.time() - t0
    
    best_rf = search.best_estimator_
    print(f"⏱️ [{tag}] RF(residual) search: {t_resid/60:.1f} min")
    print(f"✅ [{tag}] RF best params: {search.best_params_}")
    print(f"✅ [{tag}] RF best R² (CV): {search.best_score_:.6f}")

    # 테스트 세트: RF 예측 (표준화 residual) -> 원스케일 복원
    resid_pred_test_s = best_rf.predict(X_test).astype(np.float32)
    y_pred_hybrid = (base_pred_test_s + resid_pred_test_s) * y_std + y_mean
    final_mse = mean_squared_error(y_test, y_pred_hybrid)
    final_r2  = r2_score(y_test, y_pred_hybrid)
    print(f"🏁 [{tag}] Hybrid — MSE:{final_mse:.6f} R²:{final_r2:.6f}")

    artifacts = {
        "linear_model": linear_model,
        "rf_residual":  best_rf,
        "rf_best_params": search.best_params_,
        "metrics": {
            "linear_only": {"mse": float(base_mse), "r2": float(base_r2)},
            "hybrid":      {"mse": float(final_mse), "r2": float(final_r2)}
        },
        "target_scaler": {"mean": y_mean, "std": y_std, "standardized": bool(normalize_y)},
    }
    return artifacts

# ─────────────────────────────────────────────────────────
# A) Y0: Gamma / Cx / Cy (per-gray)
# ─────────────────────────────────────────────────────────        
def train_Y0_models(dataset, save_dir, patterns=('W','R','G','B')):
    components = ["Gamma", "Cx", "Cy"]
    for comp in components:
        print(f"\n=== Train Y0: {comp} ===")
        # 1) 데이터 구축
        X_all, y_all, groups = dataset.build_per_gray_y0(component=comp, patterns=patterns)  # (PK*|patterns|*256, Dx), (..,)

        # # 2) 그룹 벡터 (드롭 前 길이와 동일하게)
        # rows_per_pk = len(patterns) * 256
        # groups_full = []
        # for s in dataset.samples:
        #     groups_full.extend([s['pk']] * rows_per_pk)
        # groups_full = np.asarray(groups_full, dtype=np.int64)

        # # 3) 라벨 NaN 방어 (드롭 방식 권장)
        # mask = np.isfinite(y_all)  # True=유효
        # X_all = X_all[mask].astype(np.float32)
        # y_all = y_all[mask].astype(np.float32)
        # groups = groups_full[mask]  # ←★ 여기 꼭 같이 마스킹

        # 4) 학습
        artifacts = train_hybrid_regressor(X_all, y_all, groups, tag=f"Y0-{comp}")

        # 5) 저장
        payload = {
            "target": {"type": "Y0-per-gray", "component": comp, "patterns": patterns},
            **artifacts,
            "feature_schema": {
                "desc": "R/G/B Low/High at gray + panel_onehot + frame_rate + model_year + gray_norm + pattern_onehot",
                "channels": ['R_Low','R_High','G_Low','G_High','B_Low','B_High'],
                "add_gray_norm": True,
                "add_pattern_onehot": True
            }
        }
        path = os.path.join(save_dir, f"hybrid_{comp}_model.pkl")
        joblib.dump(payload, path, compress=("gzip", 3))
        print(f"📁 saved: {path}")

# ─────────────────────────────────────────────────────────
# B) Y1: nor.Lv slope
# ─────────────────────────────────────────────────────────
def train_Y1_model(dataset, save_dir, patterns=('W',), use_full_range=True):
    print("\n=== Train Y1 ===")
    # 1) 전체 slope 데이터셋 생성
    X_all, y_all = dataset.build_per_gray_y1(patterns=patterns, use_segment_features=True, low_only=True)  # (#PK*|patterns|*255, Dx)

    num_pk = len(dataset.samples)
    segments_per_pat = 255 # 0~254
    # 세그먼트 인덱스 벡터 (PK*패턴 수 만큼 반복)
    g_indices = np.tile(np.arange(segments_per_pat), num_pk * len(patterns))
    
    # 2) 전체 구간 OR 특정 구간 선택
    if use_full_range:
        g0, g1 = 0, 254
        mask = np.ones_like(g_indices, dtype=bool)
    else:
        g0, g1 = 88, 231
        mask = (g_indices >= g0) & (g_indices <= g1)

    X_all = X_all[mask].astype(np.float32)
    y_all = np.nan_to_num(y_all[mask], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 3) 그룹 벡터(PK 단위) 생성
    rows_per_pk = (g1 - g0 + 1) * len(patterns)
    groups = np.repeat([s['pk'] for s in dataset.samples], rows_per_pk).astype(np.int64)

    # 4) 학습
    artifacts = train_hybrid_regressor(X_all, y_all, groups, tag="Y1-slope")

    #5) 저장
    payload = {
        "target": {"type": "Y1-slope", "segments": [g0, g1], "patterns": patterns},
        **artifacts
    }
    path = os.path.join(save_dir, "hybrid_Y1_slope_model_low_only.pkl")
    joblib.dump(payload, path, compress=("gzip", 3))
    print(f"📁 saved: {path}")


# ─────────────────────────────────────────────────────────
# C) Y2: Δu′v′ (12 Macbeth 패치, 패치 one-hot 추가해 한 모델)
# ─────────────────────────────────────────────────────────
def train_Y2_model(dataset, save_dir):
    print("\n=== Train Y2 (Δu′v′ for 12 Macbeth patches, single model with patch one-hot) ===")
    MACBETH_LIST = [
        "Red","Green","Blue","Cyan","Magenta","Yellow",
        "White","Gray","Darkskin","Lightskin","Asian","Western"
    ]
    num_patches = len(MACBETH_LIST)

    # 원 데이터: meta(+lut summary) 기반 X, Δu′v′ y
    X_base, y_base = dataset.build_y2_macbeth(use_lut_summary=True)  # (#PK*12, Dx)
    X_base = X_base.astype(np.float32)
    y_base = np.nan_to_num(y_base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 패치 one-hot 추가 (build_y2_macbeth의 순서를 가정: PK 순회, 각 PK당 12행)
    patch_onehots = []
    for _ in dataset.samples:
        for i in range(num_patches):
            v = np.zeros(num_patches, np.float32)
            v[i] = 1.0
            patch_onehots.append(v)
    patch_onehots = np.vstack(patch_onehots).astype(np.float32)

    X_all = np.hstack([X_base, patch_onehots]).astype(np.float32)
    y_all = y_base

    # 그룹 벡터(PK 단위)
    groups = []
    for s in dataset.samples:
        groups.extend([s['pk']] * num_patches)
    groups = np.array(groups, dtype=np.int64)

    artifacts = train_hybrid_regressor(X_all, y_all, groups, tag="Y2-delta_uv")

    payload = {
        "target": {"type": "Y2-delta_uv", "macbeth": MACBETH_LIST},
        **artifacts,
        "feature_note": "use_lut_summary + 12-dim patch one-hot appended"
    }
    path = os.path.join(save_dir, "hybrid_Y2_delta_uv_model.pkl")
    joblib.dump(payload, path, compress=("gzip", 3))
    print(f"📁 saved: {path}")
    
def main():
    # -------------------------------------------
    # 1) 학습에 사용할 PK 리스트와 bypass PK 설정    
    # -------------------------------------------
    full_pk_range = list(range(1411, 2455))
    exclude_pks = [1934, 2154] # 학습 제외 PKs
    pk_list = [pk for pk in full_pk_range if pk not in exclude_pks]
    print(f"▶ TEST with {len(pk_list)} PKs")

    # -------------------------------------------
    # 2) 데이터셋 생성
    # -------------------------------------------
    dataset = VACDataset(pk_list)
    
    # -------------------------------------------
    # 3) 저장 경로
    # -------------------------------------------
    save_dir = os.path.dirname(__file__)
    
    # -------------------------------------------
    # 4) 학습 실행 (Y0 → Y1 → Y2)
    # -------------------------------------------
    train_Y0_models(dataset, save_dir, patterns=('W',))
    # train_Y1_model(dataset, save_dir, patterns=('W',), use_full_range=True)
    # train_Y2_model(dataset, save_dir)

    print("\n✅ ALL DONE.")
    
    # # ================== Dataset 검증용 출력 ==================
    # print("TEST")
    # pd.set_option('display.max_columns', None)
    # test_pk = [500]
    # dataset = VACDataset(test_pk)

    # X_mat, y_vec = dataset.build_per_gray_y1(patterns=('W',), use_segment_features=True)

    # print(f"\n[PK=500 Y1 Dataset Preview]")
    # print(f"X_mat shape: {X_mat.shape}")   # (255, Dx)
    # print(f"y_vec shape: {y_vec.shape}")   # (255,)
    # print("\n--- X_mat (first 3 rows) ---")
    # print(pd.DataFrame(X_mat[:3]))         # 앞부분 일부 확인
    # print("\n--- y_vec (first 10 values) ---")
    # print(y_vec[:10])

if __name__ == "__main__":
    main()
