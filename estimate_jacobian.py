# estimate_jacobian.py
# ------------------------------------------------------------
# 그레이별 자코비안 J_g 추정 스크립트 (White 패턴, High 3채널)
# X = [ΔR_H, ΔG_H, ΔB_H, panel_onehot..., frame_rate, model_year, gray_norm, LUT_j] 
# Y = [ΔCx, ΔCy, ΔGamma]
# 해법: 가중 리지 최소자승  J = (X^T W X + λI)^{-1} X^T W Y
# ------------------------------------------------------------
import os
import sys
import time
import numpy as np
import pandas as pd
import datetime

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../module/scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))    # .../module
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.VAC_dataset import VACDataset

def parse_pks(spec: str):
    """
    PK 파서: "a-b", "!a", "!a-b" 같은 표현 지원
    - "a": 단일 PK
    - "a-b": 범위
    - "!a", "!a-b": 제외
    
    예:
      "2456-2677,!2456" -> [2457, 2458, ..., 2677]
      "2457,2459-2461"  -> [2457,2459,2460,2461]
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    include = set()
    exclude = set()

    for tok in tokens:
        is_excl = tok.startswith("!")
        if is_excl:
            tok = tok[1:].strip()
            if not tok:
                continue

        # 범위 또는 단일
        if "-" in tok:
            a_str, b_str = tok.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            lo, hi = min(a, b), max(a, b)
            rng = range(lo, hi + 1)
            if is_excl:
                exclude.update(rng)
            else:
                include.update(rng)
        else:
            v = int(tok)
            if is_excl:
                exclude.add(v)
            else:
                include.add(v)

    result = sorted(p for p in include if p not in exclude)
    return result

def build_white_X_Y3(pk_list, ref_pk):
    """
    X, Y3, groups, idx_gray, ds 를 반환
    - X : (N, 12) feature (ΔR_H, ΔG_H, ΔB_H, panel_onehot, frame_rate, model_year, gray_norm, LUT_j)
    - Y3 : (N, 3)  = [dCx, dCy, dGamma]
    - groups : (N,) pk ID
    - idx_gray : X에서 gray_norm 컬럼 인덱스
    - ds : VACDataset 인스턴스
    """
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_pk)

    # component별로 한 번씩 빌드
    X_cx, y_cx, g_cx = ds.build_white_y0_delta(component='dCx')
    X_cy, y_cy, g_cy = ds.build_white_y0_delta(component='dCy')
    X_ga, y_ga, g_ga = ds.build_white_y0_delta(component='dGamma')

    # gray_norm 컬럼 위치 (VACDataset 피처 정의 기준)
    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])  # panel one-hot 길이
    idx_gray = 3 + K + 2  # [ΔR,ΔG,ΔB]=3 + panel(K) + frame_rate + model_year

    def make_dict(X, y, groups, name):
        """
        (pk, gray) -> (X_row, y_val) 또는 y_val 만 저장하는 dict 구성
        """
        gray_norm = X[:, idx_gray]
        gray_idx  = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

        dX = {}
        dY = {}
        for i in range(len(y)):
            key = (int(groups[i]), int(gray_idx[i]))  # (pk, gray)
            # X는 첫 컴포넌트(dCx)에서만 저장, 나머지는 y만 저장
            if name == "dCx":
                if key not in dX:
                    dX[key] = X[i].copy()
            dY[key] = float(y[i])
        return dX, dY

    dict_X, dict_cx = make_dict(X_cx, y_cx, g_cx, name="dCx")
    _,       dict_cy = make_dict(X_cy, y_cy, g_cy, name="dCy")
    _,       dict_ga = make_dict(X_ga, y_ga, g_ga, name="dGamma")

    # 세 컴포넌트 모두 존재하는 (pk,gray) 교집합만
    keys_common = sorted(
        set(dict_X.keys()) &
        set(dict_cx.keys()) &
        set(dict_cy.keys()) &
        set(dict_ga.keys())
    )
    if not keys_common:
        raise RuntimeError("세 컴포넌트(dCx/dCy/dGamma)의 (pk,gray) 교집합이 비어 있습니다.")

    X_rows, Y_rows, group_rows = [], [], []
    for (pk, gray) in keys_common:
        # gray 구간 필터: 2~253 (양 끝은 자코비안 크게 의미 없음 / Gamma NaN 등)
        if gray < 2 or gray > 253:
            continue

        x_row = dict_X[(pk, gray)]
        y_row = np.array([
            dict_cx[(pk, gray)],
            dict_cy[(pk, gray)],
            dict_ga[(pk, gray)]
        ], dtype=np.float32)

        # 타깃이 모두 finite일 때만 사용
        if not np.isfinite(y_row).all():
            continue

        X_rows.append(x_row)
        Y_rows.append(y_row)
        group_rows.append(pk)

    if not X_rows:
        raise RuntimeError("유효한 (pk,gray) 샘플이 없습니다. NaN 필터/gray 범위를 확인하세요.")

    X = np.vstack(X_rows).astype(np.float32)
    Y3 = np.vstack(Y_rows).astype(np.float32)
    groups = np.asarray(group_rows, dtype=np.int64)

    return X, Y3, groups, idx_gray, ds


def solve_weighted_ridge(X, Y, lam=1e-3, w=None):
    """
    가중 리지 최소자승
    (X^T W X + lam I)^{-1} X^T W Y  계산
    X: (n, d), Y: (n, k), w: (n,) or None
    반환: (coef, A)
      - coef: (d, k)
      - A   : (d, d)
    """
    if w is not None:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        sw = np.sqrt(w).reshape(-1, 1)
        Xw = X * sw
        Yw = Y * sw
    else:
        Xw, Yw = X, Y

    d = X.shape[1]
    XtX = Xw.T @ Xw
    XtY = Xw.T @ Yw
    A = XtX + lam * np.eye(d, dtype=X.dtype)

    try:
        coef = np.linalg.solve(A, XtY)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(A) @ XtY

    return coef, A


def estimate_jacobians_per_gray(pk_list, ref_pk, lam=1e-3,
                                delta_window=None, gauss_sigma=None,
                                min_samples=3):
    """
    gray별 자코비안 추정 : 각 그레이 g=0..255에 대해 J_g(3x3) 추정
    - 입력은 VACDataset 기반 스윕 데이터
    - delta_window: ||Δx|| ≤ window 필터 (선택)
    - gauss_sigma: 가우시안 가중치 σ (선택), w = exp(-||Δx||^2 / σ^2)
    
    반환:
      jac: dict[g] = { "J":(3,3), "n":n_samples, "cond":condition_number }
      df : CSV로 저장하기 편한 DataFrame
    """
    X, Y3, groups, idx_gray, ds = build_white_X_Y3(pk_list, ref_pk)

    # 디자인행렬: 첫 3열이 ΔR_H, ΔG_H, ΔB_H
    dRGB = X[:, :3]
    gray_norm = X[:, idx_gray]
    gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    # |Δx| 기반 magnitude 필터
    if delta_window is not None:
        mag = np.linalg.norm(dRGB, axis=1)
        mask_mag = mag <= float(delta_window)
    else:
        mag = np.linalg.norm(dRGB, axis=1)
        mask_mag = np.ones(len(dRGB), dtype=bool)

    jac, rows = {}, []
    for g in range(256):
        m = (gray_idx == g) & mask_mag
        n = int(m.sum())
        if n < min_samples:
            continue

        Xg = dRGB[m, :]   # (n,3)
        Yg = Y3[m, :]     # (n,3)

        # 가우시안 가중치 (Δx 크기 기반)
        if gauss_sigma is not None and gauss_sigma > 0:
            w = np.exp(-(np.linalg.norm(Xg, axis=1) ** 2) / (gauss_sigma ** 2))
        else:
            w = None

        J, A = solve_weighted_ridge(Xg, Yg, lam=lam, w=w)  # J:(3,3) 열=R,G,B / 행= Cx,Cy,Gamma
        cond = np.linalg.cond(A)

        jac[g] = {"J": J.astype(np.float32), "n": n, "cond": float(cond)}

        rows.append({
            "gray": g,
            "n_samples": n,
            "cond": float(cond),
            "J_Cx_R": float(J[0, 0]), "J_Cx_G": float(J[0, 1]), "J_Cx_B": float(J[0, 2]),
            "J_Cy_R": float(J[1, 0]), "J_Cy_G": float(J[1, 1]), "J_Cy_B": float(J[1, 2]),
            "J_Gam_R": float(J[2, 0]), "J_Gam_G": float(J[2, 1]), "J_Gam_B": float(J[2, 2]),
        })

    df = pd.DataFrame(rows).sort_values("gray")
    return jac, df

def make_default_paths(ref_pk, lam, delta_window, gauss_sigma):
    """
    출력 파일 경로 자동 생성
    """
    os.makedirs("artifacts", exist_ok=True)
    tag = f"ref{ref_pk}_lam{lam}"
    if delta_window is not None:
        tag += f"_dw{float(delta_window)}"
    if gauss_sigma is not None:
        tag += f"_gs{float(gauss_sigma)}"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join("artifacts", f"jacobians_white_high_{tag}_{ts}.csv")
    out_npy = os.path.join("artifacts", f"jacobian_bundle_{tag}_{ts}.npy")
    return out_csv, out_npy


def main():
    start_time = time.time()
    # ========================================================================================= #
    #                                         변수 지정                                        
    # ========================================================================================= #
    # 1. Sweep 데이터로 사용할 PK(s) 리스트
    pks = "2456-2677,!2456"
    pk_list = parse_pks(pks)
    
    # 2. 기준(reference) LUT PK
    ref_pk = 2582
    
    # 3. 리지 정규화 계수
    lam = 1e-3
    
    # 4. delta-window
    delta_window = 50
    
    # 5. gauss-sigma
    gauss_sigma = 30
    
    # 6. min_samples
    min_samples = 3
    # ========================================================================================= #

    jac, df = estimate_jacobians_per_gray(
        pk_list=pk_list, 
        ref_pk=ref_pk, 
        lam=lam, 
        delta_window=delta_window, 
        gauss_sigma=gauss_sigma,
        min_samples=min_samples
        )
    out_csv, out_npy = make_default_paths(ref_pk=ref_pk, lam=lam, delta_window=delta_window, gauss_sigma=gauss_sigma)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    
    # NPY 저장 (J 번들)
    J_dense = np.full((256, 3, 3), np.nan, dtype=np.float32)
    n_arr   = np.zeros(256, dtype=np.int32)
    condArr = np.full(256, np.nan, dtype=np.float32)
    for g, payload in jac.items():
        J_dense[g, :, :] = payload["J"]
        n_arr[g] = int(payload["n"])
        condArr[g] = float(payload["cond"])

    bundle = {
        "J": J_dense,
        "n": n_arr,
        "cond": condArr,
        "ref_pk": ref_pk,
        "lam": 1e-3,
        "delta_window": 50,
        "gauss_sigma": 30,
        "pk_list": pk_list,
        "gray_used": [2, 253],
        "schema": "J[gray, out(Cx,Cy,Gamma), in(R_High,G_High,B_High)]",
    }
    np.save(out_npy, bundle, allow_pickle=True)

    print(f"[OK] CSV saved -> {out_csv}")
    print(f"[OK] NPY saved -> {out_npy}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO]  {out_npy}")

    # 미리보기
    for g in (0, 32, 128, 255):
        if 0 <= g < 256 and np.isfinite(J_dense[g]).any():
            print(f"\n[g={g}] n={n_arr[g]}, cond={condArr[g]:.2e}")
            print(J_dense[g])
        else:
            print(f"\n[g={g}] no estimate (NaN or insufficient samples)")
    
    
if __name__ == "__main__":
    main()
