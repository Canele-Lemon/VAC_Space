import numpy as np
from src.modeling.VAC_dataset import VACDataset

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