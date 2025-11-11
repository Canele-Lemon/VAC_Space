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
    ref_pk  = 2744                           # 기준 LUT PK
    pk_list = [2744, 2779, 2780, 2790]       # 비교할 PK 리스트 (원하는대로 수정)
    pattern = "W"                            # 'W', 'R', 'G', 'B' 중 선택
    L       = 256
    grays   = np.arange(L, dtype=int)

    print(f"[DEBUG] ref_pk={ref_pk}, pk_list={pk_list}, pattern={pattern}")
    print("각 PK에 대해 dCx/dCy/dGamma vs gray 엑셀 + 그래프 생성 중 ...")

    # ------------------------------------------------------------
    # 1) 기본 DataFrame (gray만)
    # ------------------------------------------------------------
    df = pd.DataFrame({"gray": grays})
    # 나중에 gray>=2만 남기기 위해 우선 0~255 전체 채움

    # legend용 vac_version 보관
    pk_to_vac_version = {}

    # ------------------------------------------------------------
    # 2) 각 pk마다 prepare_Y로 dCx/dCy/dGamma 불러와서 DataFrame에 붙이기
    # ------------------------------------------------------------
    for pk in pk_list:
        builder = VACOutputBuilder(pk=pk, ref_pk=ref_pk)

        # VAC Version for legend
        try:
            vac_version, _, _ = builder.load_set_info_pk_data(pk)
        except Exception:
            # 혹시라도 실패하면 PK 번호로라도 표시
            vac_version = f"PK_{pk}"
        pk_to_vac_version[pk] = vac_version

        # ref_pk 기준 ΔY 계산
        Y = builder.prepare_Y(
            add_y0=True, add_y1=False, add_y2=False,
            y0_patterns=(pattern,)
        )

        if "Y0" not in Y or pattern not in Y["Y0"]:
            print(f"[WARN] PK={pk}: pattern={pattern} 데이터 없음 → 스킵")
            continue

        y0 = Y["Y0"][pattern]
        dCx = np.asarray(y0["dCx"], dtype=np.float32)
        dCy = np.asarray(y0["dCy"], dtype=np.float32)
        dG  = np.asarray(y0["dGamma"], dtype=np.float32)

        # 각 PK별로 컬럼 이름을 달리해서 붙인다.
        df[f"dCx_{pk}"]    = dCx
        df[f"dCy_{pk}"]    = dCy
        df[f"dGamma_{pk}"] = dG

    # ------------------------------------------------------------
    # 3) gray 0,1 제외
    # ------------------------------------------------------------
    df = df[df["gray"] >= 2].reset_index(drop=True)
    nrows = len(df)

    # ------------------------------------------------------------
    # 4) 엑셀 임시파일 생성 + 그래프 (dCx / dCy / dGamma)
    # ------------------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_path = tmp_file.name

    with pd.ExcelWriter(tmp_path, engine="xlsxwriter") as writer:
        sheet_name = "debug_dY"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]

        # 컬럼 인덱스
        col_gray = df.columns.get_loc("gray")

        # 공통 X축 (gray)
        categories = [sheet_name, 1, col_gray, nrows, col_gray]

        # --------------------------------------------------------
        # helper: component별 chart 생성
        # --------------------------------------------------------
        def make_chart_for_component(comp: str, title: str):
            """
            comp: 'dCx', 'dCy', 'dGamma'
            title: chart 제목
            """
            chart = workbook.add_chart({'type': 'line'})
            chart.set_title({'name': f'{title} vs gray (pattern={pattern})'})
            chart.set_x_axis({'name': 'gray'})
            chart.set_y_axis({'name': title})

            # pk별로 series 추가
            for pk in pk_list:
                col_name = f"{comp}_{pk}"
                if col_name not in df.columns:
                    continue  # 데이터 없는 pk는 스킵

                col_idx = df.columns.get_loc(col_name)
                vac_version = pk_to_vac_version.get(pk, f"PK_{pk}")

                chart.add_series({
                    'name': str(vac_version),      # 범례에 VAC Version 표시
                    'categories': categories,
                    'values': [sheet_name, 1, col_idx, nrows, col_idx],
                })
            return chart

        chart_dcx = make_chart_for_component("dCx", "dCx")
        chart_dcy = make_chart_for_component("dCy", "dCy")
        chart_dg  = make_chart_for_component("dGamma", "dGamma")

        # 시트 오른쪽에 세로로 배치
        base_col = df.shape[1] + 2
        worksheet.insert_chart(1,  base_col, chart_dcx)
        worksheet.insert_chart(20, base_col, chart_dcy)
        worksheet.insert_chart(39, base_col, chart_dg)

    # ------------------------------------------------------------
    # 5) 엑셀 자동 열기
    # ------------------------------------------------------------
    webbrowser.open(f"file://{tmp_path}")
    print(f"[DONE] 엑셀 파일 열기: {tmp_path}")


if __name__ == "__main__":
    main()