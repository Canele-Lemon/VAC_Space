# estimate_jacobian.py
# ------------------------------------------------------------
# 그레이별 자코비안 J_g 추정 스크립트 (White 패턴, High 3채널)
# X = [ΔR_H, ΔG_H, ΔB_H, panel_onehot(5), frame_rate, model_year, gray_norm, LUT_j] 
# Y = [ΔCx, ΔCy, ΔGamma]
# 해법: 가중 리지 최소자승  J = (X^T W X + λI)^{-1} X^T W Y
# ------------------------------------------------------------
import os
import sys
import time
import numpy as np
import pandas as pd
import datetime

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.modeling.VAC_dataset import VACDataset

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

def build_white_X_Y0(pk_list, ref_pk):
    """
    X, Y0, groups, idx_gray, ds 를 반환
    - X : (N, 12) feature (ΔR_H, ΔG_H, ΔB_H, panel_onehot, frame_rate, model_year, gray_norm, LUT_j)
    - Y0 : (N, 3)  = [dCx, dCy, dGamma]
    - groups : (N,) pk ID
    - idx_gray : X에서 gray_norm 컬럼 인덱스
    - ds : VACDataset 인스턴스
    """
    ds = VACDataset(pk_list=pk_list, ref_pk=ref_pk)

    # component별로 한 번씩 빌드
    X_cx, y_cx, g_cx = ds._build_XY0_for_jacobian_g(component='dCx')
    X_cy, y_cy, g_cy = ds._build_XY0_for_jacobian_g(component='dCy')
    X_ga, y_ga, g_ga = ds._build_XY0_for_jacobian_g(component='dGamma')

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
    Y0 = np.vstack(Y_rows).astype(np.float32)
    groups = np.asarray(group_rows, dtype=np.int64)

    return X, Y0, groups, idx_gray, ds


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
    X, Y0, groups, idx_gray, ds = build_white_X_Y0(pk_list, ref_pk)

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
        Yg = Y0[m, :]     # (n,3)

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

def debug_deltaG_sample_with_dataset(
    pk_list,
    ref_pk,
    jacobian_npy_path: str,
    target_gray: int = 128,
    target_dG: float = 50.0,
    tol_dG: float = 2.0,
    tol_RB: float = 5.0,
    max_show: int = 5,
):
    """
    VACDataset 안에서 ΔG_H ≈ target_dG 인 샘플을 찾아,
    실제 ΔY와 자코비안 예측을 비교하는 디버그 함수.

    - target_gray : 보고 싶은 gray (예: 128)
    - target_dG   : 찾고 싶은 ΔG_H 값 (예: +50)
    - tol_dG      : ΔG_H 허용 오차
    - tol_RB      : '거의 G만 건드린' 샘플을 찾기 위한 ΔR_H, ΔB_H 허용 오차
    """

    # 1) 데이터셋 로드 (자코비안 학습 때와 동일한 pk_list, ref_pk 사용)
    ds = VACDataset(pk_list=pk_list, ref_pk=ref_pk)

    # 2) 자코비안 번들 로드
    bundle  = np.load(jacobian_npy_path, allow_pickle=True).item()
    J_dense = bundle["J"]   # (256,3,3)

    if not np.isfinite(J_dense[target_gray]).any():
        print(f"[DEBUG] gray={target_gray} 에 대한 J가 NaN 입니다.")
        return

    Jg = J_dense[target_gray]  # (3,3)

    found = 0

    # 3) 각 pk / gray에서 조건에 맞는 샘플 찾기
    for s in ds.samples:
        pk = s["pk"]
        Xd = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
        Yd = s["Y"]  # {"Y0": {...}, ...}

        lut_delta = Xd["lut_delta_raw"]
        dR_arr = np.asarray(lut_delta["R_High"], dtype=np.float32)
        dG_arr = np.asarray(lut_delta["G_High"], dtype=np.float32)
        dB_arr = np.asarray(lut_delta["B_High"], dtype=np.float32)

        Y0W = Yd["Y0"]['W']
        dCx_arr   = np.asarray(Y0W["dCx"],    dtype=np.float32)
        dCy_arr   = np.asarray(Y0W["dCy"],    dtype=np.float32)
        dGam_arr  = np.asarray(Y0W["dGamma"], dtype=np.float32)

        g = int(target_gray)

        # 범위 체크
        if g < 0 or g >= 256:
            continue

        dR = float(dR_arr[g])
        dG = float(dG_arr[g])
        dB = float(dB_arr[g])

        # 조건: ΔG_H ≈ target_dG, ΔR_H/ΔB_H는 거의 0 (G만 건드린 샘플에 가깝게)
        if (abs(dG - target_dG) <= tol_dG and
            abs(dR) <= tol_RB and
            abs(dB) <= tol_RB):

            dY_real = np.array([
                float(dCx_arr[g]),
                float(dCy_arr[g]),
                float(dGam_arr[g]),
            ], dtype=np.float32)

            # 실제 ΔX 벡터에 대한 예측
            dX_real = np.array([dR, dG, dB], dtype=np.float32)
            dY_pred_real = Jg @ dX_real

            # "순수" ΔX_test = [0, target_dG, 0] 에 대한 예측도 같이 확인
            dX_pure  = np.array([0.0, target_dG, 0.0], dtype=np.float32)
            dY_pred_pure = Jg @ dX_pure

            print("\n========================================")
            print(f"[DEBUG] pk={pk}, gray={g}")
            print(f"ΔX_real  = [ΔR, ΔG, ΔB] = ({dR:+.3f}, {dG:+.3f}, {dB:+.3f})")
            print(f"ΔY_real  = [dCx, dCy, dGamma] = ({dY_real[0]:+.6f}, {dY_real[1]:+.6f}, {dY_real[2]:+.6f})")
            print(f"J · ΔX_real  (pred)           = ({dY_pred_real[0]:+.6f}, {dY_pred_real[1]:+.6f}, {dY_pred_real[2]:+.6f})")

            print(f"\n[DEBUG] 순수 ΔX_test = [0, {target_dG}, 0] 일 때 예측")
            print(f"J · [0,{target_dG},0] (pred)  = ({dY_pred_pure[0]:+.6f}, {dY_pred_pure[1]:+.6f}, {dY_pred_pure[2]:+.6f})")

            found += 1
            if found >= max_show:
                break

    if found == 0:
        print(f"[DEBUG] 조건에 맞는 샘플이 없습니다. "
              f"(gray={target_gray}, ΔG≈{target_dG}±{tol_dG}, ΔR/ΔB≈0±{tol_RB})")
        
def debug_print_XY_at_grays(pk_list, ref_pk, grays=(0, 1, 64, 128, 192, 254, 255), max_rows_per_gray=5):
    """
    지정한 gray들에 대해 X/Y 행을 그대로 출력한다.
    - max_rows_per_gray: 각 gray에서 최대 몇 개 행 출력할지 (너무 많이 찍히는 것 방지)
    """
    X, Y0, groups, idx_gray, ds = build_white_X_Y0(pk_list, ref_pk)

    gray_norm = X[:, idx_gray]
    gray_idx  = np.clip(np.round(gray_norm * 255).astype(int), 0, 255)

    print("\n================ DEBUG: X/Y rows at selected grays ================")
    print(f"ref_pk={ref_pk}, pk_list_size={len(pk_list)}, total_rows={len(X)}")
    print(f"idx_gray={idx_gray}")
    print(f"target grays={list(grays)}")
    print("-------------------------------------------------------------------")

    for g in grays:
        m = (gray_idx == int(g))
        n = int(m.sum())
        print(f"\n[gray={g}] rows={n}")

        if n == 0:
            continue

        # 너무 많이 출력되는 것 방지
        idxs = np.where(m)[0][:max_rows_per_gray]

        for i in idxs:
            pk = int(groups[i])

            x_row = X[i]
            y_row = Y0[i]

            # 보기 편하게: ΔRGB(앞 3개) / 나머지 feature는 길이가 길 수 있으니 일부만
            dR, dG, dB = float(x_row[0]), float(x_row[1]), float(x_row[2])
            dCx, dCy, dGam = float(y_row[0]), float(y_row[1]), float(y_row[2])

            print(f"  - row={i}, pk={pk}")
            print(f"    X[0:3] (dR,dG,dB) = ({dR:+.3f}, {dG:+.3f}, {dB:+.3f})")
            print(f"    X(full) = {np.array2string(x_row, precision=4, floatmode='fixed')}")
            print(f"    Y (dCx,dCy,dGamma) = ({dCx:+.6f}, {dCy:+.6f}, {dGam:+.6f})")

    print("\n===================================================================\n")

def main():
    start_time = time.time()
    # ========================================================================================= #
    #                                         변수 지정                                        
    # ========================================================================================= #
    # 1. Sweep 데이터로 사용할 PK(s) 리스트
    # pks = "2743-3002,!2743,!2744,!2984"
    # pk_list = parse_pks(pks)
    pk_list = list(range(3009, 3141))
    
    # 2. 기준(reference) LUT PK
    ref_pk = 3008
    
    # 3. 리지 정규화 계수
    lam = 1e-3
    
    # 4. delta-window
    delta_window = 80
    
    # 5. gauss-sigma
    gauss_sigma = None
    
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
        "pk_list": pk_list,
        "lam": lam,
        "delta_window": delta_window,
        "gauss_sigma": gauss_sigma,
        "gray_used": [2, 253],
        "exclude_gray_for_cxcy": [0, 5],
        "schema": "J[gray, out(Cx,Cy,Gamma), in(R_High,G_High,B_High)]",
    }
    np.save(out_npy, bundle, allow_pickle=True)

    print(f"[OK] CSV saved -> {out_csv}")
    print(f"[OK] NPY saved -> {out_npy}")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[INFO] elapsed = {elapsed:.2f} sec, bundle = {out_npy}")

    # 미리보기
    for g in (0, 32, 128, 255):
        if 0 <= g < 256 and np.isfinite(J_dense[g]).any():
            print(f"\n[g={g}] n={n_arr[g]}, cond={condArr[g]:.2e}")
            print(J_dense[g])
        else:
            print(f"\n[g={g}] no estimate (NaN or insufficient samples)")
    
    jac_path = r"artifacts\jacobian_bundle_ref2744_lam0.001_dw900.0_gs30.0_20251110_105631.npy"

    debug_deltaG_sample_with_dataset(
        pk_list=pk_list,
        ref_pk=ref_pk,
        jacobian_npy_path=jac_path,
        target_gray=128,   # sanity test 하고 싶은 gray
        target_dG=50.0,    # ΔG_H = +50 근처
        tol_dG=2.0,        # ΔG_H 허용 오차
        tol_RB=5.0,        # R/B는 거의 안 건드린 샘플만 보도록
        max_show=3,
    )
    
    #################################################
    # 데이터 확인용 디버깅
    #################################################
    # ref_pk=3008
    # pk_list=[3157]
    # debug_print_XY_at_grays(
    #     pk_list=pk_list,
    #     ref_pk=ref_pk,
    #     grays=(0, 1, 2, 32, 64, 128, 192, 253, 254, 255),
    #     max_rows_per_gray=5
    # )

    
    
if __name__ == "__main__":
    main()
