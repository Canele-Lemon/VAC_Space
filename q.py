def _debug_dump_predicted_Y0W(self, y_pred: dict, *, tag: str = "", save_csv: bool = True):
    """
    예측된 'W' 패턴 256포인트 (Gamma, Cx, Cy)를 로그로 요약 + (옵션) CSV 저장

    Parameters
    ----------
    y_pred : {"Gamma": (256,), "Cx": (256,), "Cy": (256,)}
    tag    : 로그/파일명 식별용 태그 (예: "iter1_INX_60_Y26")
    save_csv : True면 임시 CSV 파일로 저장 후 경로 로깅
    """
    import numpy as np, pandas as pd, tempfile, os, logging

    # 안전 가드
    req_keys = ("Gamma", "Cx", "Cy")
    if not all(k in y_pred for k in req_keys):
        logging.warning(f"[Predict/Debug] y_pred keys invalid: {list(y_pred.keys())}")
        return

    g = np.asarray(y_pred["Gamma"], dtype=np.float32)
    cx= np.asarray(y_pred["Cx"],    dtype=np.float32)
    cy= np.asarray(y_pred["Cy"],    dtype=np.float32)

    # ── 1) 통계 요약 로그
    def _stat(a, name):
        with np.errstate(invalid="ignore"):
            logging.debug(f"[Predict/Debug] {name}: "
                          f"shape={a.shape}, mean={np.nanmean(a):.6g}, std={np.nanstd(a):.6g}, "
                          f"min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}")
    _stat(g, "Gamma")
    _stat(cx,"Cx")
    _stat(cy,"Cy")

    # ── 2) 특정 인덱스 원소 출력 (0,1,2,127,128,129,254,255)
    idx_probe = [0,1,2,127,128,129,254,255]
    for i in idx_probe:
        if 0 <= i < 256:
            logging.debug(f"[Predict/Debug] g={i:3d} | Gamma={g[i]!r:>12} | Cx={cx[i]:.6f} | Cy={cy[i]:.6f}")

    # ── 3) (옵션) CSV 저장
    if save_csv:
        df = pd.DataFrame({"Gamma": g, "Cx": cx, "Cy": cy})
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in str(tag))
        with tempfile.NamedTemporaryFile(prefix=f"y0W_{safe_tag}_", suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
            df.to_csv(f.name, index_label="Gray")
            csv_path = f.name
        logging.info(f"[Predict/Debug] Y0(W) 256pts saved → {csv_path}")