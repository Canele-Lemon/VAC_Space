import numpy as np
import logging

def _solve_delta_rgb_for_gray(
    self,
    g: int,
    d_targets: dict,
    lam: float = 1e-3,
    wCx: float = 0.5,
    wCy: float = 0.5,
    wG:  float = 1.0,
):
    """
    주어진 gray g에서, 현재 ΔY = [dCx, dCy, dGamma]를
    자코비안 J_g를 이용해 줄이기 위한 ΔX = [ΔR_H, ΔG_H, ΔB_H]를 푼다.

    관계식:  ΔY_new ≈ ΔY + J_g · ΔX
    우리가 원하는 건 ΔY_new ≈ 0 이므로, J_g · ΔX ≈ -ΔY 를 풀어야 함.

    리지 가중 최소자승:
        argmin_ΔX || W (J_g ΔX + ΔY) ||^2 + λ ||ΔX||^2
        → (J^T W^2 J + λI) ΔX = - J^T W^2 ΔY
    """
    Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3,3)
    if not np.isfinite(Jg).all():
        logging.warning(f"[BATCH CORR] g={g}: J_g has NaN/inf → skip")
        return None

    dCx_g = float(d_targets["Cx"][g])
    dCy_g = float(d_targets["Cy"][g])
    dG_g  = float(d_targets["Gamma"][g])
    dy = np.array([dCx_g, dCy_g, dG_g], dtype=np.float32)  # (3,)

    # 이미 거의 0이면 굳이 보정 안 해도 됨
    if np.all(np.abs(dy) < 1e-6):
        return None

    # 가중치
    w_vec = np.array([wCx, wCy, wG], dtype=np.float32)     # (3,)
    WJ = w_vec[:, None] * Jg   # (3,3)
    Wy = w_vec * dy            # (3,)

    A = WJ.T @ WJ + float(lam) * np.eye(3, dtype=np.float32)  # (3,3)
    b = - WJ.T @ Wy                                           # (3,)

    try:
        dX = np.linalg.solve(A, b).astype(np.float32)
    except np.linalg.LinAlgError:
        dX = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

    dR, dG, dB = float(dX[0]), float(dX[1]), float(dX[2])
    logging.debug(
        f"[BATCH CORR] g={g}: dCx={dCx_g:+.6f}, dCy={dCy_g:+.6f}, dG={dG_g:+.6f} → "
        f"ΔR_H={dR:+.3f}, ΔG_H={dG:+.3f}, ΔB_H={dB:+.3f}"
    )
    return dR, dG, dB