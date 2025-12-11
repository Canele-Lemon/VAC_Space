    def export_measure_data_to_csv(
        self,
        pk_list=None,
        parameters=None,
        components=('Lv', 'Cx', 'Cy'),
        normalize_lv_flag: bool = True,
        with_chart: bool = False,
        open_after: bool = True,
    ):
        """
        _load_measure_data()를 이용해 측정 데이터를 내보내는 통합 헬퍼.

        사용 모드 2가지:

        1) with_chart=False  (기본값)
           - pk_list 각각에 대해 _load_measure_data 결과를 CSV(1파일/PK)로 저장
           - 디버그용 원본 테이블 보기용

        2) with_chart=True
           - pk_list (여러 개 가능)에 대해,
             'VAC_Gamma_W_Gray____' + 'VAC_GammaLinearity_60_W_Gray____' 데이터를
             한 개의 Excel(xlsx)에서 PK별 시트로 저장하고,
             각 시트에 Nor.Lv(측면) 라인 차트까지 포함.
           - 이 모드는 내부에서 parameters/components를 자동으로 설정합니다.
        """
        import tempfile
        import webbrowser
        import os
        import pandas as pd
        import numpy as np

        # pk_list 정규화
        if pk_list is None:
            pk_list = [self.pk]
        elif isinstance(pk_list, int):
            pk_list = [pk_list]
        else:
            pk_list = list(pk_list)

        # -------------------------
        # ❶ CSV only 모드 (기본)
        # -------------------------
        if not with_chart:
            if parameters is None:
                raise ValueError(
                    "[export_measure_data_to_csv] with_chart=False 모드에서는 "
                    "'parameters' 인자가 필요합니다.\n"
                    "예: parameters=['VAC_Gamma_W_Gray____', 'VAC_GammaLinearity_60_W_Gray____']"
                )

            for pk in pk_list:
                df = self._load_measure_data(
                    pk=pk,
                    parameters=parameters,
                    components=components,
                    normalize_lv_flag=normalize_lv_flag,
                )

                if df is None or df.empty:
                    print(f"[export_measure_data_to_csv] PK={pk} 에 대해 로드된 데이터가 없습니다.")
                    continue

                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"_VAC_measure_pk{pk}.csv"
                )
                tmp_path = tmp.name
                tmp.close()

                df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
                print(f"[OK] measure data CSV saved → {tmp_path}")

                if open_after:
                    webbrowser.open(f"file://{os.path.abspath(tmp_path)}")

            return  # CSV 모드 종료

        # -------------------------
        # ❷ Excel + chart 모드
        #    (옛 load_multiple_pk_data_with_chart 통합)
        # -------------------------
        # 이 모드는 W 패턴 정면/측면 Gamma 데이터 전용으로 설계
        #   - 정면:  VAC_Gamma_W_Gray____
        #   - 측면:  VAC_GammaLinearity_60_W_Gray____
        front_param = "VAC_Gamma_W_Gray____"
        side_param  = "VAC_GammaLinearity_60_W_Gray____"

        # 임시 엑셀 파일 생성
        tmp_xlsx = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        xlsx_path = tmp_xlsx.name
        tmp_xlsx.close()

        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            for pk in pk_list:
                # ---- 정면 W Gamma 측정 데이터 (정규화 Lv 포함) ----
                df_front = self._load_measure_data(
                    pk=pk,
                    parameters=[front_param],
                    components=('Lv', 'Cx', 'Cy'),
                    normalize_lv_flag=True,  # Nor.Lv로 쓰기 위해 True
                )

                # ---- 측면 60도 W GammaLinearity 데이터 (Nor.Lv) ----
                df_side = self._load_measure_data(
                    pk=pk,
                    parameters=[side_param],
                    components=('Lv',),
                    normalize_lv_flag=True,  # 여기서도 Nor.Lv로 사용
                )

                sheet_name = f"PK_{pk}"

                # --------- 시트에 front/side 테이블 쓰기 ---------
                start_row_side = 0

                if df_front is not None and not df_front.empty:
                    # 정면: Pattern_Window='W' 인 것만 pivot
                    sub_f = df_front[
                        (df_front["Pattern_Window"] == "W")
                        & df_front["Component"].isin(["Lv", "Cx", "Cy"])
                    ].copy()

                    if not sub_f.empty:
                        front_pivot = (
                            sub_f
                            .pivot(index="Gray_Level", columns="Component", values="Data")
                            .reset_index()
                            .rename(columns={"Gray_Level": "Gray"})
                        )
                        front_pivot.to_excel(
                            writer, sheet_name=sheet_name,
                            startrow=0, index=False
                        )
                        start_row_side = len(front_pivot) + 3  # 아래에 side 테이블 배치

                if df_side is not None and not df_side.empty:
                    # 측면: Pattern_Window='W', Component='Lv' 만 사용 (이미 Nor.Lv)
                    sub_s = df_side[
                        (df_side["Pattern_Window"] == "W")
                        & (df_side["Component"] == "Lv")
                    ].copy()

                    if not sub_s.empty:
                        sub_s = sub_s.sort_values("Gray_Level")
                        side_df = sub_s[["Gray_Level", "Data"]].rename(
                            columns={"Gray_Level": "Gray", "Data": "Nor. Lv"}
                        )
                        side_df.to_excel(
                            writer, sheet_name=sheet_name,
                            startrow=start_row_side, index=False
                        )

                        # --------- Nor. Lv 차트 추가 ---------
                        worksheet = writer.sheets[sheet_name]
                        chart = workbook.add_chart({"type": "line"})

                        # 엑셀 내에서의 컬럼 위치 계산
                        col_gray = side_df.columns.get_loc("Gray")      # 보통 0
                        col_nlv  = side_df.columns.get_loc("Nor. Lv")   # 보통 1

                        first_row = start_row_side + 1     # 헤더 아래부터
                        last_row  = start_row_side + len(side_df)

                        chart.add_series({
                            "name": "Nor. Lv (60deg, W)",
                            "categories": [
                                sheet_name, first_row, col_gray,
                                last_row,  col_gray
                            ],
                            "values": [
                                sheet_name, first_row, col_nlv,
                                last_row,  col_nlv
                            ],
                        })
                        chart.set_title({"name": "Side View Nor. Lv (W, 60deg)"})
                        chart.set_x_axis({"name": "Gray"})
                        chart.set_y_axis({"name": "Nor. Lv"})

                        worksheet.insert_chart(0, 10, chart)

        print(f"[OK] Excel with charts saved → {xlsx_path}")

        if open_after:
            webbrowser.open(f"file://{os.path.abspath(xlsx_path)}")