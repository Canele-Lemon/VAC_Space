def export_mapped_lut_to_csv(self, pk=None, ref_pk=None, out_path=None, open_after=True):
    """
    LUT_index_mapping.csv의 매핑 j(0..4095)를 사용해
    Gray i(0..255)별로 j에서의 LUT 값을 CSV로 저장합니다.

    포함 컬럼:
      Gray, LUT_j,
      <ch>_tgt, <ch>_ref, <ch>_delta   for ch in [R_Low, R_High, G_Low, G_High, B_Low, B_High]

    Parameters
    ----------
    pk : int | None
        내보낼 target VAC_SET_Info PK (None이면 self.pk 유지)
    ref_pk : int | None
        비교할 reference VAC_SET_Info PK (None이면 delta/참조 없이 target만 기록)
    out_path : str | None
        저장 경로 (None이면 ./artifacts/mapped_lut_<pk>[_ref<ref_pk>]_<timestamp>.csv)
    open_after : bool
        저장 후 파일을 기본 앱으로 열지 여부
    """
    import pandas as pd
    import numpy as np
    import os, datetime, webbrowser
    os.makedirs("artifacts", exist_ok=True)

    if pk is not None:
        self.pk = int(pk)

    # 매핑 j 불러오기
    j_map = self._load_lut_index_mapping()  # (256,) int32

    # 타깃/레퍼런스 4096 LUT 불러오기
    lut4096_tgt = self._load_vacdata_lut4096(self.pk)
    if lut4096_tgt is None:
        raise RuntimeError(f"[export] No VAC_Data for target VAC_SET_Info.PK={self.pk}")

    lut4096_ref = None
    if ref_pk is not None:
        lut4096_ref = self._load_vacdata_lut4096(int(ref_pk))
        if lut4096_ref is None:
            raise RuntimeError(f"[export] No VAC_Data for reference VAC_SET_Info.PK={ref_pk}")

    channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']

    rows = []
    for i in range(256):  # gray 0..255
        j = int(j_map[i])
        row = {"Gray": i, "LUT_j": j}
        for ch in channels:
            v_tgt = float(lut4096_tgt[ch][j]) if lut4096_tgt is not None else np.nan
            row[f"{ch}_tgt"] = v_tgt
            if lut4096_ref is not None:
                v_ref = float(lut4096_ref[ch][j])
                row[f"{ch}_ref"] = v_ref
                row[f"{ch}_delta"] = v_tgt - v_ref
        rows.append(row)

    df = pd.DataFrame(rows)

    # 컬럼 정렬(가독성)
    ordered_cols = ["Gray", "LUT_j"]
    for ch in channels:
        ordered_cols += [f"{ch}_tgt"]
        if lut4096_ref is not None:
            ordered_cols += [f"{ch}_ref", f"{ch}_delta"]
    df = df[ordered_cols]

    # 저장 경로
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_path is None:
        tag = f"mapped_lut_pk{self.pk}"
        if ref_pk is not None:
            tag += f"_ref{int(ref_pk)}"
        out_path = os.path.join("artifacts", f"{tag}_{ts}.csv")

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV saved → {out_path}")

    if open_after:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")

    return out_path