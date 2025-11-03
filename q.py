# estimate_jacobian.py
# ------------------------------------------------------------
# 그레이별 자코비안 J_g 추정 스크립트 (White 패턴, High 3채널)
# X = [ΔR_H, ΔG_H, ΔB_H, panel_onehot..., frame_rate, model_year, gray_norm, LUT_j]
# Y = [ΔCx, ΔCy, ΔGamma]
# 해법: 가중 리지 최소자승  J = (X^T W X + λI)^{-1} X^T W Y
# ------------------------------------------------------------

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

# --- 프로젝트 루트 기준으로 import 경로 보정 ---
CUR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CUR, ".."))   # 필요 시 조정
if ROOT not in sys.path:
    sys.path.append(ROOT)

# VAC_dataset.py의 실제 위치에 맞춰 경로 수정하세요.
# 예) module/scripts/VAC_dataset.py 라면 아래처럼:
try:
    from scripts.VAC_dataset import VACDataset
except Exception:
    # fallback: 같은 폴더에 있다면
    from VAC_dataset import VACDataset


def build_white_X_Y3(pk_list, ref_pk):
    """
    동일한 순서의 X에 대해 dCx/dCy/dGamma를 세 열로 쌓아 (X, Y3) 반환
    """
    ds = VACDataset(pk_list=pk_list, ref_vac_info_pk=ref_pk)

    X_cx, y_cx, g_cx = ds.build_white_y0_delta(component='dCx')
    X_cy, y_cy, g_cy = ds.build_white_y0_delta(component='dCy')
    X_ga, y_ga, g_ga = ds.build_white_y0_delta(component='dGamma')

    # 안전성 체크
    if not (X_cx.shape == X_cy.shape == X_ga.shape):
        raise RuntimeError("X 행렬 형태가 일치하지 않습니다 (dCx/dCy/dGamma).")
    if not (np.all(g_cx == g_cy) and np.all(g_cx == g_ga)):
        raise RuntimeError("groups(패널 식별) 순서가 일치하지 않습니다.")

    # gray_norm 위치 계산
    K = len(ds.samples[0]["X"]["meta"]["panel_maker"])  # panel one-hot 길이
    idx_gray = 3 + K + 2  # [ΔR,ΔG,ΔB]=3 + panel(K) + frame(1) + year(1) => gray_norm

    Y3 = np.stack([y_cx, y_cy, y_ga], axis=1).astype(np.float32)
    X = X_cx.astype(np.float32)
    return X, Y3, g_cx, idx_gray, ds


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
    main()