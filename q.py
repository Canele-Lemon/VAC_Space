# debug_check.py
import re
import numpy as np
import pandas as pd
import tempfile
import webbrowser

from src.data_preparation.prepare_output import VACOutputBuilder


def get_vac_version(pk: int, ref_pk: int) -> str:
    """
    pk에 대한 VAC Version 문자열을 가져온다.
    예: 'G+50', 'R+50_G+50' 등
    """
    builder = VACOutputBuilder(pk=pk, ref_pk=ref_pk)
    vac_version, _, _ = builder.load_set_info_pk_data(pk)
    return str(vac_version)


def parse_dlut_from_vac_version(vac_version: str) -> float | None:
    """
    vac_version 문자열에서 숫자만 추출해서 ΔLUT로 사용.
    예:
      'G+50'         -> 50
      'R-100'        -> -100
      'R+50_G+50'    -> 50  (첫 번째 숫자 기준)
    필요하면 여기 로직을 원하는 대로 바꿀 수 있음.
    """
    nums = re.findall(r'[-+]?\d+', vac_version)
    if not nums:
        return None
    return float(nums[0])


def draw_dY_vs_gray(ref_pk: int, pk_list: list[int], pattern: str = "W"):
    """
    여러 PK에 대해:
      - ref_pk 기준 dCx, dCy, dGamma (pattern=W/R/G/B 선택)
      - gray(2~255) vs dY 그래프 (엑셀 + 라인 그래프 3개)
      - 범례는 VAC Version
    """
    L = 256
    grays = np.arange(L, dtype=int)

    print(f"[dY_vs_gray] ref_pk={ref_pk}, pk_list={pk_list}, pattern={pattern}")

    df = pd.DataFrame({"gray": grays})
    pk_to_vac_version = {}

    # 각 PK에 대해 dY 계산
    for pk in pk_list:
        builder = VACOutputBuilder(pk=pk, ref_pk=ref_pk)

        try:
            vac_version = get_vac_version(pk, ref_pk)
        except Exception:
            vac_version = f"PK_{pk}"
        pk_to_vac_version[pk] = vac_version

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

        df[f"dCx_{pk}"]    = dCx
        df[f"dCy_{pk}"]    = dCy
        df[f"dGamma_{pk}"] = dG

    # gray 0,1 제거
    df = df[df["gray"] >= 2].reset_index(drop=True)
    nrows = len(df)

    # 엑셀 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_path = tmp_file.name

    with pd.ExcelWriter(tmp_path, engine="xlsxwriter") as writer:
        sheet_name = "dY_vs_gray"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]

        col_gray = df.columns.get_loc("gray")
        categories = [sheet_name, 1, col_gray, nrows, col_gray]

        def make_chart_for_component(comp: str, title: str):
            chart = workbook.add_chart({'type': 'line'})
            chart.set_title({'name': f'{title} vs gray (pattern={pattern})'})
            chart.set_x_axis({'name': 'gray'})
            chart.set_y_axis({'name': title})

            for pk in pk_list:
                col_name = f"{comp}_{pk}"
                if col_name not in df.columns:
                    continue
                col_idx = df.columns.get_loc(col_name)
                vac_version = pk_to_vac_version.get(pk, f"PK_{pk}")

                chart.add_series({
                    'name': str(vac_version),
                    'categories': categories,
                    'values': [sheet_name, 1, col_idx, nrows, col_idx],
                })
            return chart

        chart_dcx = make_chart_for_component("dCx",    "dCx")
        chart_dcy = make_chart_for_component("dCy",    "dCy")
        chart_dg  = make_chart_for_component("dGamma", "dGamma")

        base_col = df.shape[1] + 2
        worksheet.insert_chart(1,  base_col, chart_dcx)
        worksheet.insert_chart(20, base_col, chart_dcy)
        worksheet.insert_chart(39, base_col, chart_dg)

    webbrowser.open(f"file://{tmp_path}")
    print(f"[dY_vs_gray] 엑셀 파일 열림: {tmp_path}")


def draw_dY_vs_dLUT(
    ref_pk: int,
    pk_list: list[int],
    pattern: str = "W",
    gray: int = 128,
):
    """
    여러 PK에 대해:
      - ref_pk 기준 dCx/dCy/dGamma (pattern=W/R/G/B, 특정 gray에서)
      - x축: vac_version 문자열에서 추출한 ΔLUT 숫자
      - y축: dCx / dCy / dGamma
      - 각 점은 한 PK (legend = vac_version)
    """
    print(f"[dY_vs_dLUT] ref_pk={ref_pk}, pk_list={pk_list}, pattern={pattern}, gray={gray}")

    records = []

    for pk in pk_list:
        builder = VACOutputBuilder(pk=pk, ref_pk=ref_pk)

        try:
            vac_version = get_vac_version(pk, ref_pk)
        except Exception:
            vac_version = f"PK_{pk}"

        dlut = parse_dlut_from_vac_version(vac_version)
        if dlut is None:
            print(f"[WARN] PK={pk}, vac_version={vac_version}: ΔLUT 숫자 파싱 실패 → 스킵")
            continue

        Y = builder.prepare_Y(
            add_y0=True, add_y1=False, add_y2=False,
            y0_patterns=(pattern,)
        )
        if "Y0" not in Y or pattern not in Y["Y0"]:
            print(f"[WARN] PK={pk}: pattern={pattern} 데이터 없음 → 스킵")
            continue

        y0 = Y["Y0"][pattern]
        dCx_arr = np.asarray(y0["dCx"], dtype=np.float32)
        dCy_arr = np.asarray(y0["dCy"], dtype=np.float32)
        dG_arr  = np.asarray(y0["dGamma"], dtype=np.float32)

        if not (0 <= gray < len(dCx_arr)):
            print(f"[WARN] PK={pk}: gray={gray} 범위 밖 → 스킵")
            continue

        dCx_g = float(dCx_arr[gray])
        dCy_g = float(dCy_arr[gray])
        dG_g  = float(dG_arr[gray])

        if not (np.isfinite(dCx_g) and np.isfinite(dCy_g) and np.isfinite(dG_g)):
            print(f"[WARN] PK={pk}, gray={gray}: dY 중 NaN/inf 존재 → 스킵")
            continue

        records.append({
            "pk": pk,
            "vac_version": vac_version,
            "dLUT": dlut,
            "dCx": dCx_g,
            "dCy": dCy_g,
            "dGamma": dG_g,
        })

    if not records:
        print("[dY_vs_dLUT] 유효한 데이터가 없습니다.")
        return

    df = pd.DataFrame(records).sort_values("dLUT").reset_index(drop=True)
    nrows = len(df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_path = tmp_file.name

    with pd.ExcelWriter(tmp_path, engine="xlsxwriter") as writer:
        sheet_name = "dY_vs_dLUT"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]

        col_dlut = df.columns.get_loc("dLUT")
        col_dcx  = df.columns.get_loc("dCx")
        col_dcy  = df.columns.get_loc("dCy")
        col_dg   = df.columns.get_loc("dGamma")

        def make_scatter_for_component(col_y: int, title: str):
            chart = workbook.add_chart({'type': 'scatter'})
            chart.set_title({'name': f'{title} vs ΔLUT (pattern={pattern}, gray={gray})'})
            chart.set_x_axis({'name': 'ΔLUT'})
            chart.set_y_axis({'name': title})

            # 각 PK를 하나의 series로, 점 하나씩
            for i, row in df.iterrows():
                r = i + 1  # 헤더 한 줄 때문에
                vac_version = str(row["vac_version"])

                chart.add_series({
                    'name':       vac_version,
                    'categories': [sheet_name, r + 0, col_dlut, r + 0, col_dlut],
                    'values':     [sheet_name, r + 0, col_y,    r + 0, col_y],
                    'marker': {'type': 'automatic'},
                })

            return chart

        chart_dcx = make_scatter_for_component(col_dcx, "dCx")
        chart_dcy = make_scatter_for_component(col_dcy, "dCy")
        chart_dg  = make_scatter_for_component(col_dg,  "dGamma")

        base_col = df.shape[1] + 2
        worksheet.insert_chart(1,  base_col, chart_dcx)
        worksheet.insert_chart(20, base_col, chart_dcy)
        worksheet.insert_chart(39, base_col, chart_dg)

    webbrowser.open(f"file://{tmp_path}")
    print(f"[dY_vs_dLUT] 엑셀 파일 열림: {tmp_path}")


def main():
    ref_pk   = 2744
    pattern  = "W"

    # 1) dY vs gray 용 PK 리스트 (원하는대로 수정)
    pk_list_gray = [2744, 2779, 2780, 2790]

    # 2) dLUT vs dY 용 PK 리스트 (요청하신 리스트)
    pk_list_dlut = [2745, 2747, 2751, 2775,
                    2757, 2758, 2761, 2763,
                    2765, 2769, 2771, 2773]

    # gray 2~255에 대한 dY vs gray
    draw_dY_vs_gray(ref_pk=ref_pk, pk_list=pk_list_gray, pattern=pattern)

    # gray=128에서 dLUT vs dY
    draw_dY_vs_dLUT(ref_pk=ref_pk, pk_list=pk_list_dlut,
                    pattern=pattern, gray=128)


if __name__ == "__main__":
    main()