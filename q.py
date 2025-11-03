# estimate_jacobian.py
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../module/scripts
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))    # .../module
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.VAC_dataset import VACDataset

# ------------------------------------------------------------
# 그레이별 자코비안 J_g 추정 스크립트 (White 패턴, High 3채널)
# X = [ΔR_H, ΔG_H, ΔB_H, panel_onehot..., frame_rate, model_year, gray_norm, LUT_j]
# Y = [ΔCx, ΔCy, ΔGamma]
# 해법: 가중 리지 최소자승  J = (X^T W X + λI)^{-1} X^T W Y
# ------------------------------------------------------------

def build_white_X_Y3(pk_list, ref_pk):
    """
    세 컴포넌트(dCx, dCy, dGamma)의 (pk, gray) 교집합만 남겨
    X(공통), Y3=[dCx,dCy,dGamma]를 같은 순서로 정렬해서 반환.
    """
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_pk)

    X_cx, y_cx, g_cx = ds.build_white_y0_delta(component='dCx')
    X_cy, y_cy, g_cy = ds.build_white_y0_delta(component='dCy')
    X_ga, y_ga, g_ga = ds.build_white_y0_delta(component='dGamma')

    # 공통: gray_norm 위치 계산
    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])
    idx_gray_cx = 3 + K + 2
    idx_gray_cy = 3 + K + 2
    idx_gray_ga = 3 + K + 2

    def keys_from(X, groups, idx_gray):
        gray_norm = X[:, idx_gray]
        gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)
        # 키 = (pk, gray)
        return np.stack([groups.astype(np.int64), gray_idx.astype(np.int64)], axis=1)

    keys_cx = keys_from(X_cx, g_cx, idx_gray_cx)
    keys_cy = keys_from(X_cy, g_cy, idx_gray_cy)
    keys_ga = keys_from(X_ga, g_ga, idx_gray_ga)

    # 각 키를 문자열로 만들어 집합 교집합
    def key_str(keys):
        return np.char.add(keys[:,0].astype(str), ":" + keys[:,1].astype(str))

    kc = key_str(keys_cx)
    ky = key_str(keys_cy)
    kg = key_str(keys_ga)

    common = np.intersect1d(np.intersect1d(kc, ky), kg)
    if common.size == 0:
        raise RuntimeError("세 컴포넌트의 (pk,gray) 교집합이 비어 있습니다.")

    # 공통 키의 인덱스 선택자를 만든다
    def indexer(all_keys_str, common_keys):
        # 공통 키 순서에 맞춰 인덱스 배열 생성
        lookup = {k:i for i,k in enumerate(all_keys_str)}
        return np.array([lookup[k] for k in common_keys], dtype=np.int64)

    idx_cx = indexer(kc, common)
    idx_cy = indexer(ky, common)
    idx_ga = indexer(kg, common)

    # 동일 행 순서로 정렬
    X = X_cx[idx_cx].astype(np.float32)          # X는 어떤 컴포넌트에서 가져와도 동일
    y_cx_sel = y_cx[idx_cx].astype(np.float32)
    y_cy_sel = y_cy[idx_cy].astype(np.float32)
    y_ga_sel = y_ga[idx_ga].astype(np.float32)

    # g=2..253만 사용 + 세 타깃 모두 finite
    gray_norm = X[:, 3 + K + 2]
    gray_idx  = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)
    Y3 = np.stack([y_cx_sel, y_cy_sel, y_ga_sel], axis=1)

    core_mask = (gray_idx >= 2) & (gray_idx <= 253) & np.isfinite(Y3).all(axis=1)
    X   = X[core_mask]
    Y3  = Y3[core_mask]
    # groups는 pk, gray에서 pk만 필요하면 pk만 복원
    # pk는 common 키에서 앞부분이므로 재구성
    groups = np.array([int(k.split(":")[0]) for k in common], dtype=np.int64)[core_mask]

    # gray_norm 컬럼 인덱스 반환 (나중에 사용할 수 있도록)
    idx_gray = 3 + K + 2
    return X, Y3, groups, idx_gray, ds


def solve_weighted_ridge(X, Y, lam=1e-3, w=None):
    """
    (X^T W X + lam I)^{-1} X^T W Y  계산
    X: (n, d), Y: (n, k), w: (n,) or None
    반환: (d, k)
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

    # 상태 나쁠 때 대비
    try:
        coef = np.linalg.solve(A, XtY)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(A) @ XtY
    return coef, A


def estimate_jacobians_per_gray(pk_list, ref_pk, lam=1e-3,
                                delta_window=None, gauss_sigma=None,
                                min_samples=3):
    """
    각 그레이 g=0..255에 대해 J_g(3x3) 추정
    - 입력은 VACDataset 기반 스윕 데이터
    - delta_window: ||Δx|| ≤ window 필터 (선택)
    - gauss_sigma: 가우시안 가중치 σ (선택), w = exp(-||Δx||^2 / σ^2)
    """
    X, Y3, groups, idx_gray, ds = build_white_X_Y3(pk_list, ref_pk)

    # 디자인행렬: 첫 3열이 ΔR_H, ΔG_H, ΔB_H
    dRGB = X[:, :3]
    gray_norm = X[:, idx_gray]
    gray_idx = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    # 선택적 마그니튜드 필터
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

        Xg = dRGB[m, :]  # (n,3)
        Yg = Y3[m, :]    # (n,3)

        # 가우시안 가중치(선택)
        if gauss_sigma is not None and gauss_sigma > 0:
            w = np.exp(-(np.linalg.norm(Xg, axis=1) ** 2) / (gauss_sigma ** 2))
        else:
            w = None

        J, A = solve_weighted_ridge(Xg, Yg, lam=lam, w=w)  # J:(3,3) 열=R,G,B / 행= Cx,Cy,Gamma
        cond = np.linalg.cond(A)

        jac[g] = {"J": J.astype(np.float32), "n": n, "cond": float(cond)}

        # CSV 한 줄 (행= Cx,Cy,Gamma / 열= R,G,B)
        rows.append({
            "gray": g,
            "n_samples": n,
            "cond": float(cond),
            "J_Cx_R": float(J[0,0]), "J_Cx_G": float(J[0,1]), "J_Cx_B": float(J[0,2]),
            "J_Cy_R": float(J[1,0]), "J_Cy_G": float(J[1,1]), "J_Cy_B": float(J[1,2]),
            "J_Gam_R": float(J[2,0]), "J_Gam_G": float(J[2,1]), "J_Gam_B": float(J[2,2]),
        })

    df = pd.DataFrame(rows).sort_values("gray")
    return jac, df


def main():
    ap = argparse.ArgumentParser(description="그레이별 자코비안 Jg 추정 (White, High 채널)")
    ap.add_argument("--pks", type=str, required=True,
                    help="스윕에 사용된 VAC_SET_Info PK 목록(콤마 구분). 예: 3001,3002,3003")
    ap.add_argument("--ref", type=int, required=True,
                    help="참조 VAC_Info PK (예: 2582)")
    ap.add_argument("--lam", type=float, default=1e-3, help="리지 λ")
    ap.add_argument("--delta-window", type=float, default=None,
                    help="|Δx| ≤ window 필터 (예: 50). 미지정 시 필터 없음")
    ap.add_argument("--gauss-sigma", type=float, default=None,
                    help="가우시안 가중 σ (예: 30). 미지정 시 가중치 없음")
    ap.add_argument("--min-samples", type=int, default=3,
                    help="그레이별 최소 샘플 수")
    ap.add_argument("--out-csv", type=str, default="jacobians_white_high.csv",
                    help="출력 CSV 경로")
    ap.add_argument("--out-npy", type=str, default=None,
                    help="numpy .npy(dict) 저장 경로(선택)")

    args = ap.parse_args()

    pk_list = [int(x.strip()) for x in args.pks.split(",") if x.strip()]
    jac, df = estimate_jacobians_per_gray(
        pk_list=pk_list,
        ref_pk=args.ref,
        lam=args.lam,
        delta_window=args.delta_window,
        gauss_sigma=args.gauss_sigma,
        min_samples=args.min_samples
    )

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV saved -> {args.out_csv}")
    if args.out_npy:
        # npy에는 각 g의 J(3x3), n, cond 저장
        np.save(args.out_npy, jac, allow_pickle=True)
        print(f"[OK] NPY saved -> {args.out_npy}")

    # 간단 프리뷰
    for g in (0, 32, 128, 255):
        if g in jac:
            J = jac[g]["J"]
            print(f"\n[g={g}] n={jac[g]['n']}, cond={jac[g]['cond']:.2e}")
            print(" rows=[Cx,Cy,Gamma], cols=[R_H,G_H,B_H]")
            print(J)
        else:
            print(f"\n[g={g}] no estimate (samples < min or filtered)")

if __name__ == "__main__":
    X, Y3, g_cx, idx_gray, ds = build_white_X_Y3(pk_list=[2635], ref_pk=2582)
    for i in range(100):
        print(Y3)

이렇게 했더니 이번엔 아래 에러가 떠요
Traceback (most recent call last):
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\estimate_jacobian.py", line 229, in <module>
    X, Y3, g_cx, idx_gray, ds = build_white_X_Y3(pk_list=[2635], ref_pk=2582)
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\estimate_jacobian.py", line 54, in build_white_X_Y3
    kc = key_str(keys_cx)
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\module\scripts\estimate_jacobian.py", line 52, in key_str
    return np.char.add(keys[:,0].astype(str), ":" + keys[:,1].astype(str))
numpy.core._exceptions._UFuncNoLoopError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U1'), dtype('<U21')) -> None
