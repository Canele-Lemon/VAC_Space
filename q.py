def _update_lut_chart_and_table(self, lut_dict, *, downsample_step=8):
    """
    LUT(0..4095)을 차트/테이블에 반영.
    - 차트: R/G/B × (Low/High) 6개 라인
    - 테이블: 기존 update_rgbchannel_table 재사용
    - downsample_step: 차트 표시용 다운샘플 간격 (성능/가독성 목적). 1이면 전체 4096pts 그립니다.
    """
    try:
        import numpy as np
        import pandas as pd

        required = ["R_Low", "R_High", "G_Low", "G_High", "B_Low", "B_High"]
        for k in required:
            if k not in lut_dict:
                raise KeyError(f"lut_dict missing key: {k}")

        # ---- 1) 테이블 데이터프레임 준비 (4096 그대로) ----
        df = pd.DataFrame({
            "R_Low":  np.asarray(lut_dict["R_Low"],  dtype=np.float32),
            "R_High": np.asarray(lut_dict["R_High"], dtype=np.float32),
            "G_Low":  np.asarray(lut_dict["G_Low"],  dtype=np.float32),
            "G_High": np.asarray(lut_dict["G_High"], dtype=np.float32),
            "B_Low":  np.asarray(lut_dict["B_Low"],  dtype=np.float32),
            "B_High": np.asarray(lut_dict["B_High"], dtype=np.float32),
        })

        # 길이가 4096이 아닐 경우, 4096으로 보정(선형보간)
        def _ensure_4096(arr):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.size == 4096:
                return arr
            if arr.size < 2:
                # 너무 짧으면 0으로 채움
                out = np.zeros(4096, dtype=np.float32)
                if arr.size == 1:
                    out[:] = arr[0]
                return out
            x_src = np.linspace(0, 1, arr.size)
            x_dst = np.linspace(0, 1, 4096)
            return np.interp(x_dst, x_src, arr).astype(np.float32)

        for col in df.columns:
            df[col] = _ensure_4096(df[col].values)

        # ---- 2) 차트 업데이트 ----
        chart = self.vac_optimization_lut_chart  # XYChart 인스턴스
        # 라인 메타: (열이름, 표시라벨, 색, 라인스타일)
        series_meta = [
            ("R_Low",  "R Low",  "red",   "--"),
            ("R_High", "R High", "red",   "-"),
            ("G_Low",  "G Low",  "green", "--"),
            ("G_High", "G High", "green", "-"),
            ("B_Low",  "B Low",  "blue",  "--"),
            ("B_High", "B High", "blue",  "-"),
        ]

        # 표시용 다운샘플(성능/가독성). 1이면 4096점 그대로 그림
        step = max(1, int(downsample_step))
        xs_plot = np.arange(0, 4096, step, dtype=int)

        for col, label, color, ls in series_meta:
            ys = df[col].values
            ys_plot = ys[::step]

            # 라인이 없으면 생성
            if label not in chart.lines:
                chart.add_line(key=label, color=color, linestyle=ls, axis_index=0, label=label)

            # 내부 버퍼 갱신
            chart.data[label]['x'] = xs_plot.tolist()
            chart.data[label]['y'] = ys_plot.astype(float).tolist()

            # 라인 데이터 교체
            line = chart.lines[label]
            line.set_data(chart.data[label]['x'], chart.data[label]['y'])

        # 축 리밋/뷰 갱신 + 리드로우
        for ax in chart.axes:
            ax.relim()
            ax.autoscale_view()
        chart.canvas.draw()

        # ---- 3) 테이블 업데이트 ----
        #   * 기존 메서드 그대로 활용 (여기가 여러분 UI 규격을 가장 잘 맞춤)
        self.update_rgbchannel_table(df, self.ui.vac_table_rbgLUT_4)

    except KeyError as e:
        logging.error(f"[LUT Chart] KeyError: {e}")
    except Exception as e:
        logging.exception(e)