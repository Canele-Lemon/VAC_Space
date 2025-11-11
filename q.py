# debug_check.py
import os
import sys
import re
import tempfile
import webbrowser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_preparation.prepare_output import VACOutputBuilder
from config.db_config import engine


# ---------------------------------------------------
# 1) vac_version 에서 ΔLUT 숫자 파싱
# ---------------------------------------------------
def parse_dlut_from_vac_version(vac_version: str) -> float | None:
    """
    vac_version 문자열에서 ΔLUT 값을 추출.
    예) "B+50", "R+50_G+50", "L2_G+50", "G50" 등.

    규칙:
      1) [+/-]가 붙은 숫자들 먼저 찾음 → 있으면 첫 번째를 사용
         예) "L2_G+50" -> ['+50'] → 50
      2) 없으면 부호 없는 숫자들을 찾고, 마지막 숫자를 사용
         예) "G50" -> ['50'] → 50
      3) 그래도 없으면 None
    """
    # 1) 부호가 반드시 있는 숫자 먼저
    signed_nums = re.findall(r'[+-]\d+', vac_version)
    if signed_nums:
        return float(signed_nums[0])

    # 2) 부호 없는 숫자
    plain_nums = re.findall(r'\d+', vac_version)
    if plain_nums:
        return float(plain_nums[-1])

    return None


# ---------------------------------------------------
# 2) pk → vac_version, dY(W패턴) 얻기
# ---------------------------------------------------
def get_vac_version(pk: int) -> str:
    """
    W_VAC_SET_Info 테이블에서 VAC_Version 문자열만 가져오기
    """
    query = f"""
        SELECT `VAC_Version`
        FROM `W_VAC_SET_Info`
        WHERE `PK` = {pk}
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return f"PK_{pk}"
    return str(df["VAC_Version"].iloc[0])


def get_dY_for_pk(pk: int, ref_pk: int, pattern: str = "W") -> dict:
    """
    pk와 ref_pk 기준으로
    dCx, dCy, dGamma (pattern=W)의 256 gray 배열을 반환

    return:
      {
        "dCx": (256,),
        "dCy": (256,),
        "dGamma": (256,),
      }
    """
    builder = VACOutputBuilder(pk=pk, ref_pk=ref_pk)
    y0 = builder.compute_Y0_struct(patterns=(pattern,))  # { 'W': {dGamma,dCx,dCy} }
    comp = y0[pattern]
    return {
        "dCx": comp["dCx"],
        "dCy": comp["dCy"],
        "dGamma": comp["dGamma"],
    }


# ---------------------------------------------------
# 3) dY vs gray (참고용, 기존 함수)
# ---------------------------------------------------
def draw_dY_vs_gray(pk_list, ref_pk, pattern="W"):
    """
    각 pk에 대해 dCx/dCy/dGamma vs gray (x축=gray, y축=dY)
    범례는 vac_version
    """
    # pk별 데이터 준비
    data = {}
    for pk in pk_list:
        vac_version = get_vac_version(pk)
        dY = get_dY_for_pk(pk, ref_pk=ref_pk, pattern=pattern)
        data[pk] = {
            "vac_version": vac_version,
            "dCx": dY["dCx"],
            "dCy": dY["dCy"],
            "dGamma": dY["dGamma"],
        }

    grays = np.arange(256, dtype=int)
    mask = (grays >= 2) & (grays <= 253)
    grays_use = grays[mask]

    # 엑셀 + 차트
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine="xlsxwriter") as writer:
            workbook = writer.book

            for comp_name in ("dCx", "dCy", "dGamma"):
                sheet_name = f"{comp_name}_vs_gray"
                rows = []
                for pk in pk_list:
                    vac_version = data[pk]["vac_version"]
                    arr = np.asarray(data[pk][comp_name], dtype=np.float32)
                    row = {"pk": pk, "vac_version": vac_version}
                    for g in grays_use:
                        v = arr[g]
                        row[f"g{g}"] = float(v) if np.isfinite(v) else np.nan
                    rows.append(row)

                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                ws = writer.sheets[sheet_name]

                chart = workbook.add_chart({"type": "line"})

                col_gray0 = df.columns.get_loc(f"g{grays_use[0]}")
                col_gray_last = df.columns.get_loc(f"g{grays_use[-1]}")
                n_rows = len(df)

                # pk별 series: x=gray, y=dY
                # (엑셀에서는 x축을 카테고리로 gray label을 넣는 방식)
                for pk_idx, pk in enumerate(pk_list):
                    col_start = col_gray0
                    col_end = col_gray_last
                    chart.add_series({
                        "name":       [sheet_name, pk_idx + 1, 1],  # vac_version 셀
                        "categories": [sheet_name, 0, col_gray0, 0, col_gray_last],
                        "values":     [sheet_name, pk_idx + 1, col_gray0, pk_idx + 1, col_gray_last],
                    })

                chart.set_title({"name": f"{comp_name} vs gray (pattern={pattern})"})
                chart.set_x_axis({"name": "gray"})
                chart.set_y_axis({"name": comp_name})
                chart.set_legend({"position": "bottom"})
                ws.insert_chart(0, col_gray_last + 2, chart)

        webbrowser.open(f"file://{tmp.name}")


# ---------------------------------------------------
# 4) dY vs dLUT (요청하신 부분 수정)
# ---------------------------------------------------
def draw_dY_vs_dLUT(pk_list, ref_pk, pattern="W"):
    """
    dY vs ΔLUT 그래프 그리기 (엑셀로).
      - x축: ΔLUT (vac_version에서 숫자 파싱)
      - y축: dY (dCx, dCy, dGamma)
      - 범례: Gray g (예: Gray 32, Gray 64, ..., Gray 224)

    포함 그레이: g = 32, 64, 96, 128, 160, 192, 224
    (0,1은 사용하지 않음)
    """
    grays_use = [32, 64, 96, 128, 160, 192, 224]

    # pk별 ΔLUT, dY 준비
    records = {}  # pk -> dict
    for pk in pk_list:
        vac_version = get_vac_version(pk)
        dlut = parse_dlut_from_vac_version(vac_version)

        if dlut is None:
            # ΔLUT 파싱 실패하면 스킵하거나 0으로 둘지 선택 가능
            # 여기서는 스킵
            print(f"[WARN] PK={pk}, vac_version='{vac_version}' → ΔLUT 파싱 실패, 스킵")
            continue

        dY = get_dY_for_pk(pk, ref_pk=ref_pk, pattern=pattern)
        records[pk] = {
            "vac_version": vac_version,
            "ΔLUT": dlut,
            "dCx": dY["dCx"],
            "dCy": dY["dCy"],
            "dGamma": dY["dGamma"],
        }

    if not records:
        print("[ERROR] 유효한 ΔLUT / dY 데이터가 없습니다.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        with pd.ExcelWriter(tmp.name, engine="xlsxwriter") as writer:
            workbook = writer.book

            for comp_name in ("dCx", "dCy", "dGamma"):
                sheet_name = f"{comp_name}_vs_dLUT"
                rows = []

                for pk, payload in records.items():
                    vac_version = payload["vac_version"]
                    dlut = payload["ΔLUT"]
                    arr = np.asarray(payload[comp_name], dtype=np.float32)
                    row = {
                        "pk": pk,
                        "vac_version": vac_version,
                        "ΔLUT": dlut,
                    }
                    for g in grays_use:
                        v = arr[g]
                        row[f"g{g}"] = float(v) if np.isfinite(v) else np.nan
                    rows.append(row)

                df = pd.DataFrame(rows)
                df = df.sort_values("ΔLUT")  # x축 정렬

                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                ws = writer.sheets[sheet_name]

                chart = workbook.add_chart({"type": "line"})

                col_x = df.columns.get_loc("ΔLUT")
                n_rows = len(df)

                first_data_row = 1
                last_data_row = n_rows

                # 이제 "각 gray 별"로 series 추가
                #   x: ΔLUT
                #   y: g32, g64, ... 각각
                for g in grays_use:
                    col_y = df.columns.get_loc(f"g{g}")
                    chart.add_series({
                        "name":       f"Gray {g}",
                        "categories": [sheet_name, first_data_row, col_x, last_data_row, col_x],
                        "values":     [sheet_name, first_data_row, col_y, last_data_row, col_y],
                    })

                chart.set_title({"name": f"{comp_name} vs ΔLUT (pattern={pattern})"})
                chart.set_x_axis({"name": "ΔLUT (step)", "major_gridlines": {"visible": True}})
                chart.set_y_axis({"name": comp_name})
                chart.set_legend({"position": "bottom"})

                # ΔLUT / g컬럼 뒤쪽에 차트 삽입
                ws.insert_chart(0, len(df.columns) + 2, chart)

        webbrowser.open(f"file://{tmp.name}")


# ---------------------------------------------------
# 5) main: 예시
# ---------------------------------------------------
if __name__ == "__main__":
    # 기준 LUT pk (Base / ref)
    ref_pk = 2744

    # 사용하신 dLUT sweep 대상 pk들
    pk_list = [2745, 2747, 2751, 2775, 2757, 2758, 2761, 2763, 2765, 2769, 2771, 2773]

    # 1) gray별 dY 그래프 (필요하면)
    # draw_dY_vs_gray(pk_list, ref_pk, pattern="W")

    # 2) ΔLUT별 dY 그래프 (요청하신 부분)
    draw_dY_vs_dLUT(pk_list, ref_pk, pattern="W")