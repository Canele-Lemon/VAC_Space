# debug_check.py
import numpy as np
import pandas as pd
import tempfile
import webbrowser

from src.data_preparation.prepare_output import VACOutputBuilder


def main():
    # ------------------------------------------------------------
    # 설정 값
    # ------------------------------------------------------------
    ref_pk    = 2744   # 기준 LUT (Base)
    target_pk = 2779   # 비교 대상 LUT (예: G+50)
    pattern   = 'W'    # 'W', 'R', 'G', 'B' 중 선택
    L         = 256
    grays     = np.arange(L, dtype=int)

    print(f"ref_pk={ref_pk}, target_pk={target_pk}, pattern={pattern}")
    print("dCx/dCy/dGamma vs gray 엑셀 + 그래프 생성 중 ...")

    # ------------------------------------------------------------
    # 1) 절대값 Y0_abs (각 PK 자체의 Cx/Cy/Gamma)
    # ------------------------------------------------------------
    builder_ref = VACOutputBuilder(pk=ref_pk,    ref_pk=ref_pk)
    builder_tgt = VACOutputBuilder(pk=target_pk, ref_pk=ref_pk)

    y_abs_ref = builder_ref.compute_Y0_struct_abs()
    y_abs_tgt = builder_tgt.compute_Y0_struct_abs()

    Cx_ref = y_abs_ref[pattern]["Cx"].astype(np.float32)
    Cy_ref = y_abs_ref[pattern]["Cy"].astype(np.float32)
    G_ref  = y_abs_ref[pattern]["Gamma"].astype(np.float32)

    Cx_tgt = y_abs_tgt[pattern]["Cx"].astype(np.float32)
    Cy_tgt = y_abs_tgt[pattern]["Cy"].astype(np.float32)
    G_tgt  = y_abs_tgt[pattern]["Gamma"].astype(np.float32)

    # ------------------------------------------------------------
    # 2) manual ΔY = target - ref
    # ------------------------------------------------------------
    dCx_manual = Cx_tgt - Cx_ref
    dCy_manual = Cy_tgt - Cy_ref
    dG_manual  = G_tgt  - G_ref

    # ------------------------------------------------------------
    # 3) prepare_Y() 기준 ΔY (자코비안에서 사용하는 값)
    # ------------------------------------------------------------
    builder_tgt_rel = VACOutputBuilder(pk=target_pk, ref_pk=ref_pk)
    Y_target = builder_tgt_rel.prepare_Y(
        add_y0=True, add_y1=False, add_y2=False,
        y0_patterns=(pattern,)
    )
    y0_tgt = Y_target["Y0"][pattern]

    dCx_prepare = y0_tgt["dCx"].astype(np.float32)
    dCy_prepare = y0_tgt["dCy"].astype(np.float32)
    dG_prepare  = y0_tgt["dGamma"].astype(np.float32)

    # ------------------------------------------------------------
    # 4) DataFrame 구성 (엑셀로 내보낼 데이터)
    # ------------------------------------------------------------
    df = pd.DataFrame({
        "gray": grays,

        "Cx_ref": Cx_ref,
        "Cx_tgt": Cx_tgt,
        "dCx_manual":  dCx_manual,
        "dCx_prepare": dCx_prepare,

        "Cy_ref": Cy_ref,
        "Cy_tgt": Cy_tgt,
        "dCy_manual":  dCy_manual,
        "dCy_prepare": dCy_prepare,

        "Gamma_ref": G_ref,
        "Gamma_tgt": G_tgt,
        "dGamma_manual":  dG_manual,
        "dGamma_prepare": dG_prepare,
    })

    # ------------------------------------------------------------
    # 5) 엑셀 임시파일 생성 + 그래프 (dCx / dCy / dGamma)
    # ------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_path = tmp_file.name

    with pd.ExcelWriter(tmp_path, engine="xlsxwriter") as writer:
        sheet_name = "debug"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]

        nrows = len(df)
        # 각 컬럼 인덱스
        col_gray = df.columns.get_loc("gray")

        col_dCx_m  = df.columns.get_loc("dCx_manual")
        col_dCx_p  = df.columns.get_loc("dCx_prepare")

        col_dCy_m  = df.columns.get_loc("dCy_manual")
        col_dCy_p  = df.columns.get_loc("dCy_prepare")

        col_dG_m   = df.columns.get_loc("dGamma_manual")
        col_dG_p   = df.columns.get_loc("dGamma_prepare")

        # 공통 category (x축: gray)
        categories = [sheet_name, 1, col_gray, nrows, col_gray]

        # ---------- dCx chart ----------
        chart_dcx = workbook.add_chart({'type': 'line'})
        chart_dcx.set_title({'name': f'dCx vs gray (pattern={pattern})'})
        chart_dcx.set_x_axis({'name': 'gray'})
        chart_dcx.set_y_axis({'name': 'dCx'})

        chart_dcx.add_series({
            'name': 'dCx_manual',
            'categories': categories,
            'values': [sheet_name, 1, col_dCx_m, nrows, col_dCx_m],
        })
        chart_dcx.add_series({
            'name': 'dCx_prepareY',
            'categories': categories,
            'values': [sheet_name, 1, col_dCx_p, nrows, col_dCx_p],
        })

        # ---------- dCy chart ----------
        chart_dcy = workbook.add_chart({'type': 'line'})
        chart_dcy.set_title({'name': f'dCy vs gray (pattern={pattern})'})
        chart_dcy.set_x_axis({'name': 'gray'})
        chart_dcy.set_y_axis({'name': 'dCy'})

        chart_dcy.add_series({
            'name': 'dCy_manual',
            'categories': categories,
            'values': [sheet_name, 1, col_dCy_m, nrows, col_dCy_m],
        })
        chart_dcy.add_series({
            'name': 'dCy_prepareY',
            'categories': categories,
            'values': [sheet_name, 1, col_dCy_p, nrows, col_dCy_p],
        })

        # ---------- dGamma chart ----------
        chart_dg = workbook.add_chart({'type': 'line'})
        chart_dg.set_title({'name': f'dGamma vs gray (pattern={pattern})'})
        chart_dg.set_x_axis({'name': 'gray'})
        chart_dg.set_y_axis({'name': 'dGamma'})

        chart_dg.add_series({
            'name': 'dGamma_manual',
            'categories': categories,
            'values': [sheet_name, 1, col_dG_m, nrows, col_dG_m],
        })
        chart_dg.add_series({
            'name': 'dGamma_prepareY',
            'categories': categories,
            'values': [sheet_name, 1, col_dG_p, nrows, col_dG_p],
        })

        # 차트를 시트 오른쪽에 세로로 배치
        # (row, col) = (시작 행, 시작 열)
        base_col = df.shape[1] + 2   # 데이터 오른쪽에 차트 배치
        worksheet.insert_chart(1,  base_col,     chart_dcx)
        worksheet.insert_chart(20, base_col,     chart_dcy)
        worksheet.insert_chart(39, base_col,     chart_dg)

    # ------------------------------------------------------------
    # 6) 엑셀 자동 열기
    # ------------------------------------------------------------
    webbrowser.open(f"file://{tmp_path}")
    print(f"엑셀 파일 열기: {tmp_path}")


if __name__ == "__main__":
    main()