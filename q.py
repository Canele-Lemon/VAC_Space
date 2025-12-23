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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

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
def train_Y0_models(dataset, save_dir, patterns=('W',), exclude_gray_for_cxcy=(0,5)):
    components = ["dGamma", "dCx", "dCy"]
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
    
    for comp in components:
        print(f"\n=== Train Y0: {comp} ===")
        
        # 1) 데이터 구축
        X_all, y_all, groups = dataset.build_XY_dataset(
            target="y0",
            component=comp,
            channels=channels,
            patterns=patterns,
        )

        artifacts = train_hybrid_regressor(X_all, y_all, groups, tag=f"Y0-{comp}")

        payload = {
            "target": {"type": "Y0-per-gray", "component": comp, "patterns": patterns},
            **artifacts,
            "feature_schema": {
                "desc": "ΔLUT(6ch) + meta + gray_norm + LUT_j",
                "channels": list(channels),
                "add_gray_norm": True,
                "add_LUT_j": True,
                "note": f"exclude_gray_for_cxcy={exclude_gray_for_cxcy} is applied inside VACDataset._build_XY0"
            }
        }
        path = os.path.join(save_dir, f"hybrid_{comp}_model.pkl")
        joblib.dump(payload, path, compress=("gzip", 3))
        print(f"📁 saved: {path}")

# ─────────────────────────────────────────────────────────
# B) Y1: nor.Lv slope
# ─────────────────────────────────────────────────────────
def train_Y1_model(dataset, save_dir, patterns=('W',)):
    print("\n=== Train Y1 ===")
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
    
    X_all, y_all, groups = dataset.build_XY_dataset(
    target="y1",
    channels=channels,
    patterns=patterns,
    )

    # y_all에 NaN 있는 경우 대비
    mask = np.isfinite(y_all)
    X_all = X_all[mask].astype(np.float32)
    y_all = y_all[mask].astype(np.float32)
    groups = groups[mask].astype(np.int64)
    
    artifacts = train_hybrid_regressor(X_all, y_all, groups, tag="Y1-slope")
    
    payload = {
            "target": {"type": "Y1-slope", "patterns": patterns},
            **artifacts,
            "feature_schema": {"desc": "segment-center gray ΔLUT(6ch)+meta+gray_norm+LUT_j"}
    }
    path = os.path.join(save_dir, "hybrid_Y1_slope_model_low_only.pk앗앗l")
    joblib.dump(payload, path, compress=("gzip", 3))
    print(f"📁 saved: {path}")

# ─────────────────────────────────────────────────────────
# C) Y2: Δu′v′ (12 Macbeth 패치, 패치 one-hot 추가해 한 모델)
# ─────────────────────────────────────────────────────────
def train_Y2_model(dataset, save_dir):
    print("\n=== Train Y2 (delta_uv) ===")
    channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')

    X_all, y_all, groups = dataset.build_XY_dataset(
        target="y2",
        channels=channels,
    )

    mask = np.isfinite(y_all)
    X_all = X_all[mask].astype(np.float32)
    y_all = y_all[mask].astype(np.float32)
    groups = groups[mask].astype(np.int64)

    artifacts = train_hybrid_regressor(X_all, y_all, groups, tag="Y2-delta_uv")

    payload = {
        "target": {"type": "Y2-delta_uv"},
        **artifacts,
        "feature_schema": {"desc": "3 gray-triplets ΔLUT + meta + pattern one-hot (already inside dataset)"}
    }
    path = os.path.join(save_dir, "hybrid_Y2_delta_uv_model.pkl")
    joblib.dump(payload, path, compress=("gzip", 3))
    print(f"📁 saved: {path}")
    
    
def preview(name, X, y, groups, n=3):
    print(f"\n[PREVIEW] {name}")
    print("X:", X.shape, "y:", y.shape, "groups:", groups.shape)
    print("finite y ratio:", np.isfinite(y).mean())
    print("y min/max:", np.nanmin(y), np.nanmax(y))
    print("unique groups:", len(np.unique(groups)))
    print("first rows X[0:3]:\n", X[:n, :3])
    print("first y:", y[:n], "first groups:", groups[:n])
    
def main():
    BYPASS_PK = 3007
    # # -------------------------------------------
    # # 1) 학습에 사용할 PK 리스트와 bypass PK 설정    
    # # -------------------------------------------
    # full_pk_range = list(range(1411, 2455))
    # exclude_pks = [1934, 2154] # 학습 제외 PKs
    # pk_list = [pk for pk in full_pk_range if pk not in exclude_pks]
    # print(f"▶ TEST with {len(pk_list)} PKs")

    # # -------------------------------------------
    # # 2) 데이터셋 생성
    # # -------------------------------------------
    # dataset = VACDataset(pk_list)
    
    # # -------------------------------------------
    # # 3) 저장 경로
    # # -------------------------------------------
    # save_dir = os.path.dirname(__file__)
    
    # # -------------------------------------------
    # # 4) 학습 실행 (Y0 → Y1 → Y2)
    # # -------------------------------------------
    # train_Y0_models(dataset, save_dir, patterns=('W',))
    # # train_Y1_model(dataset, save_dir, patterns=('W',), use_full_range=True)
    # # train_Y2_model(dataset, save_dir)

    # print("\n✅ ALL DONE.")
    
    # ================== Dataset 검증용 출력 ==================
    print("TEST")
    pd.set_option('display.max_columns', None)
    test_pk = [3008]
    dataset = VACDataset(test_pk, ref_pk=BYPASS_PK)

    X, y, groups = dataset.build_XY_dataset(
        target="y0",
        component="dGamma",
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W','R','G','B'),
    )

    print(f"\n[PK=3007 Y0 Dataset Preview]")
    print(f"X_mat shape: {X.shape}")   # (255, Dx)
    print(f"y_vec shape: {y.shape}")   # (255,)
    print("\n--- X_mat (first 3 rows) ---")
    print(pd.DataFrame(X[:3]))         # 앞부분 일부 확인
    print("\n--- y_vec (first 10 values) ---")
    print(y[:10])

를 실행했을 때 아래처럼 나옵니다. 

[PK=3007 Y0 Dataset Preview]
X_mat shape: (253, 15)
y_vec shape: (253,)

--- X_mat (first 3 rows) ---
     0     1     2     3     4     5    6    7    8    9    10    11    12  \
0 -14.0   7.0 -14.0   7.0 -14.0   7.0  0.0  0.0  0.0  0.0  1.0  60.0  26.0
1 -31.0  15.0 -31.0  15.0 -31.0  15.0  0.0  0.0  0.0  0.0  1.0  60.0  26.0
2 -45.0  22.0 -45.0  22.0 -45.0  22.0  0.0  0.0  0.0  0.0  1.0  60.0  26.0

         13    14
0  0.007843  16.0
1  0.011765  36.0
2  0.015686  52.0

--- y_vec (first 10 values) ---
[ 1.3256073e-04  1.5616560e-01 -9.7430944e-02  1.6355515e-04
  1.7142296e-04 -1.9010544e-02 -1.9743919e-02 -2.8309345e-02
 -2.1104097e-02 -1.6113281e-02]


6    7    8    9    10 = panal_maker (one-hot)
11 = frame_rate
12 = model_year
13 = nor_gray
14 = lut_index

이렇게 됩니다. 보시다시피 패턴은 포함되어 있지 않습니다. 포함하는게 좋을까요? W 패턴만 사용하긴 합니다만,,, 

패턴 정보를 X에 포함시킬 경우 어디를 수정하면 되는지와,

위 코드 기준 학습 실행은 어떻게 하면 되는지 알려주세요.
(ref_pk=BAPASS_PK, target_pk_list=3008~3316
