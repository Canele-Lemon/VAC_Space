# app_main.py

import os
import sys
import numpy as np
import pandas as pd
import tempfile

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QHeaderView
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt

import pyqtgraph as pg

from ui_app import Ui_MainWindow


FULL_POINTS = 4096
LOW_LUT_CSV = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\4. 기준 LUT + OFFSET\기준 LUT\LUT_low_values_SIN1300_254gray를4092로변경.csv"


class LUTEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set up table model
        self.model = QStandardItemModel(256, 5)
        self.model.setHorizontalHeaderLabels(
            ["Gray", "LUT Index", "R_High", "G_High", "B_High"]
        )
        self.ui.tableView_LUT.setModel(self.model)
        self.ui.tableView_LUT.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # create pyqtgraph plot
        self._create_plot()

        # 디폴트 Low LUT 표시
        self.load_and_plot_low_lut()

        # connect signals
        # ★ dataChanged 시그널은 (topLeft, bottomRight, roles) 인자가 오므로 *args로 받게 수정할 예정
        self.model.dataChanged.connect(self.on_table_changed)
        self.ui.actionOpen_control_points_value.triggered.connect(self.load_csv)
        self.ui.actionExpert_LUT_CSV.triggered.connect(self.save_csv)
        self.ui.actionExport_vacparam_json.triggered.connect(self.export_json)

        ### NEW: 체크박스 → 커브 표시/숨김
        # UI에 다음 이름의 체크박스가 있다고 가정:
        # ckBox_R_High, ckBox_G_High, ckBox_B_High,
        # ckBox_R_Low,  ckBox_G_Low,  ckBox_B_Low
        if hasattr(self.ui, "ckBox_R_High"):
            self.ui.ckBox_R_High.toggled.connect(self.update_curve_visibility)
        if hasattr(self.ui, "ckBox_G_High"):
            self.ui.ckBox_G_High.toggled.connect(self.update_curve_visibility)
        if hasattr(self.ui, "ckBox_B_High"):
            self.ui.ckBox_B_High.toggled.connect(self.update_curve_visibility)
        if hasattr(self.ui, "ckBox_R_Low"):
            self.ui.ckBox_R_Low.toggled.connect(self.update_curve_visibility)
        if hasattr(self.ui, "ckBox_G_Low"):
            self.ui.ckBox_G_Low.toggled.connect(self.update_curve_visibility)
        if hasattr(self.ui, "ckBox_B_Low"):
            self.ui.ckBox_B_Low.toggled.connect(self.update_curve_visibility)

        # 초기 상태에서 체크박스 상태에 맞춰 가시성 세팅
        self.update_curve_visibility()
        ### NEW 끝

        # initialize Gray column
        for i in range(256):
            item = QStandardItem(str(i))
            item.setFlags(Qt.ItemIsEnabled)  # read-only
            self.model.setItem(i, 0, item)

    # ---------------------------------------------------------
    # Graph setup
    # ---------------------------------------------------------
    def _create_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.ui.verticalLayout_RGBvalues_vs_Gray.addWidget(self.plot_widget)

        self.plot_widget.addLegend()

        self.curve_R = self.plot_widget.plot(
            pen=pg.mkPen(color=(255, 0, 0), width=2), name="R_High"
        )
        self.curve_G = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 255, 0), width=2), name="G_High"
        )
        self.curve_B = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 0, 255), width=2), name="B_High"
        )
        self.curve_R_low = self.plot_widget.plot(
            pen=pg.mkPen(color=(255, 100, 100), width=1, style=Qt.DashLine),
            name="R_Low"
        )
        self.curve_G_low = self.plot_widget.plot(
            pen=pg.mkPen(color=(100, 255, 100), width=1, style=Qt.DashLine),
            name="G_Low"
        )
        self.curve_B_low = self.plot_widget.plot(
            pen=pg.mkPen(color=(100, 100, 255), width=1, style=Qt.DashLine),
            name="B_Low"
        )

    # Low LUT 
    def load_low_lut_4096(self):
        df = pd.read_csv(LOW_LUT_CSV)

        # R/G/B Low 컬럼 자동 탐색
        def pick_col(cands):
            for c in cands:
                if c in df.columns:
                    return c
            raise ValueError("Low LUT CSV에서 R_Low/G_Low/B_Low 컬럼을 찾지 못했습니다.")

        col_r = pick_col(["R_Low", "R_low", "R"])
        col_g = pick_col(["G_Low", "G_low", "G"])
        col_b = pick_col(["B_Low", "B_low", "B"])

        Rl = df[col_r].to_numpy(dtype=float)[:4096]
        Gl = df[col_g].to_numpy(dtype=float)[:4096]
        Bl = df[col_b].to_numpy(dtype=float)[:4096]

        return Rl, Gl, Bl

    def load_and_plot_low_lut(self):
        Rl, Gl, Bl = self.load_low_lut_4096()
        x = np.arange(4096)

        self.curve_R_low.setData(x, Rl)
        self.curve_G_low.setData(x, Gl)
        self.curve_B_low.setData(x, Bl)

    # ---------------------------------------------------------
    # Load CSV
    # ---------------------------------------------------------
    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "256 Knot CSV 불러오기", "", "CSV Files (*.csv)"
        )
        if not fname:
            return

        df = pd.read_csv(fname)

        # expect columns: Gray8, Gray12, R_High, G_High, B_High
        for r in range(256):
            self.model.setItem(r, 0, QStandardItem(str(df.iloc[r]["Gray8"])))
            self.model.setItem(r, 1, QStandardItem(str(df.iloc[r]["Gray12"])))
            self.model.setItem(r, 2, QStandardItem(str(df.iloc[r]["R_High"])))
            self.model.setItem(r, 3, QStandardItem(str(df.iloc[r]["G_High"])))
            self.model.setItem(r, 4, QStandardItem(str(df.iloc[r]["B_High"])))

        # CSV 불러온 직후에도 0~4095 클램프 + 그래프 업데이트
        self.clamp_table_values()
        self.update_plot()

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------
    def save_csv(self):
        # 저장하기 전에 한 번 더 클램프
        self.clamp_table_values()

        fname, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "", "CSV Files (*.csv)")
        if not fname:
            return

        data = []
        for r in range(256):
            row = [
                int(round(float(self.model.item(r, 0).text()))),
                int(round(float(self.model.item(r, 1).text()))),
                int(round(float(self.model.item(r, 2).text()))),
                int(round(float(self.model.item(r, 3).text()))),
                int(round(float(self.model.item(r, 4).text()))),
            ]
            data.append(row)

        df = pd.DataFrame(
            data,
            columns=["Gray8", "Gray12", "R_High", "G_High", "B_High"]
        )
        df.to_csv(fname, index=False)

    def build_full_LUT_dataframe(self):
        """
        현재 테이블의 256개 High knot 값 + Low LUT(4096) + 보간 결과를 담아
        GrayLevel_window, R_Low, R_High, ... B_High 총 4096행 DataFrame 반환
        """
        # LUT 사용 전에도 안전하게 클램프
        self.clamp_table_values()

        # --- Low LUT ---
        Rl, Gl, Bl = self.load_low_lut_4096()
        j_axis = np.arange(4096)

        # --- High knot 읽기 ---
        Gray12 = []
        Rvals = []
        Gvals = []
        Bvals = []

        for r in range(256):
            item_g12 = self.model.item(r, 1)
            item_r   = self.model.item(r, 2)
            item_g   = self.model.item(r, 3)
            item_b   = self.model.item(r, 4)

            if (item_g12 is None or item_g12.text().strip() == "" or
                item_r   is None or item_r.text().strip() == "" or
                item_g   is None or item_g.text().strip() == "" or
                item_b   is None or item_b.text().strip() == ""):
                # High LUT 불완전 → None 반환
                return None

            Gray12.append(float(item_g12.text()))
            Rvals.append(float(item_r.text()))
            Gvals.append(float(item_g.text()))
            Bvals.append(float(item_b.text()))

        # numpy 변환 & sort
        Gray12 = np.array(Gray12, float)
        Rvals = np.array(Rvals, float)
        Gvals = np.array(Gvals, float)
        Bvals = np.array(Bvals, float)

        idx = np.argsort(Gray12)
        Gray12 = Gray12[idx]
        Rvals = Rvals[idx]
        Gvals = Gvals[idx]
        Bvals = Bvals[idx]

        # 보간
        R_full = np.interp(j_axis, Gray12, Rvals)
        G_full = np.interp(j_axis, Gray12, Gvals)
        B_full = np.interp(j_axis, Gray12, Bvals)

        # DataFrame 구성
        LUT = pd.DataFrame({
            "GrayLevel_window": j_axis,
            "R_Low":  Rl,
            "R_High": R_full,
            "G_Low":  Gl,
            "G_High": G_full,
            "B_Low":  Bl,
            "B_High": B_full,
        })

        for c in ["R_Low","R_High","G_Low","G_High","B_Low","B_High"]:
            LUT[c] = np.rint(LUT[c]).astype(int)
    
        return LUT

    def write_default_data(self, file):
        default_data = """{																					
"DRV_valc_major_ctrl"	:	[	0,	1	],																
"DRV_valc_pattern_ctrl_0"	:	[	5,	1	],																
"DRV_valc_pattern_ctrl_1"	:	[	[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	],	
			[	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0	],	
			[	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1	]      	 ],
"DRV_valc_sat_ctrl"	:	[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],		
"DRV_valc_hpf_ctrl_0"	:	[	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0	],		
"DRV_valc_hpf_ctrl_1"	:		1,																		
"""
        file.write(default_data)

    def write_LUT_data(self, file, LUT):
        channels = {
            "RchannelLow": 'R_Low',
            "RchannelHigh": 'R_High',
            "GchannelLow": 'G_Low',
            "GchannelHigh": 'G_High',
            "BchannelLow": 'B_Low',
            "BchannelHigh": 'B_High'
        }
        
        for i, (channel_name, col) in enumerate(channels.items()):
            file.write(f'"{channel_name}"\t:\t[\t')
            data = LUT[col].values
            reshaped_data = np.reshape(data, (256, 16)).tolist()

            for row_index, row in enumerate(reshaped_data):
                formatted_row = ',\t'.join(map(lambda x: str(int(x)), row))
                if row_index == 0:
                    file.write(f'{formatted_row},\n')
                elif row_index == len(reshaped_data) - 1:
                    file.write(f'\t\t\t{formatted_row}')
                else:
                    file.write(f'\t\t\t{formatted_row},\n')

            if i == len(channels) - 1:
                file.write("\t]\n")
            else:
                file.write("\t],\n")

        file.write("}")
        
    def export_json(self):
        LUT = self.build_full_LUT_dataframe()
        if LUT is None:
            print("LUT 데이터가 완전하지 않아 JSON 생성 불가.")
            return
        
        ###################################################
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_LUT_full4096.csv")
        tmp_path = tmp.name
        tmp.close()

        LUT.to_csv(tmp_path, index=False, encoding="utf-8-sig")

        print(f"[OK] LUT 4096 CSV 임시 저장: {tmp_path}")

        # Windows 자동 열기
        try:
            os.startfile(tmp_path)  # Windows
        except Exception as e:
            print("자동 열기 실패:", e)
        ###################################################   
         
        fname, _ = QFileDialog.getSaveFileName(
            self, "VAC Param JSON 저장", "", "JSON Files (*.json)"
        )
        if not fname:
            return

        with open(fname, "w", encoding="utf-8") as f:
            self.write_default_data(f)
            self.write_LUT_data(f, LUT)

        print(f"[OK] JSON 저장 완료 → {fname}")

    # ---------------------------------------------------------
    # LUT 테이블 값 클램프 (0 ~ 4095)
    # ---------------------------------------------------------
    def clamp_table_values(self):
        """
        테이블의 LUT Index / R_High / G_High / B_High 값을
        0 ~ 4095 범위로 강제하고, 클램프된 값은 셀에도 반영.
        """
        old_block = self.model.blockSignals(True)  # 재귀 방지

        try:
            for r in range(256):
                for c in (1, 2, 3, 4):  # LUT Index, R_High, G_High, B_High
                    item = self.model.item(r, c)
                    if item is None:
                        continue
                    txt = item.text().strip()
                    if txt == "":
                        continue
                    try:
                        v = float(txt)
                    except ValueError:
                        # 숫자 아니면 스킵
                        continue

                    v_clamped = max(0.0, min(4095.0, v))

                    if v_clamped != v:
                        item.setText(str(int(round(v_clamped))))
        finally:
            self.model.blockSignals(old_block)

    # ---------------------------------------------------------
    # When user edits R/G/B
    # ---------------------------------------------------------
    def on_table_changed(self, *args):
        # 먼저 값 클램프
        self.clamp_table_values()
        # 그 다음 그래프 업데이트
        self.update_plot()

    # ---------------------------------------------------------
    # Perform interpolation and redraw graph
    # ---------------------------------------------------------
    def update_plot(self):
        Gray12 = []
        Rvals = []
        Gvals = []
        Bvals = []

        for r in range(256):
            item_g12 = self.model.item(r, 1)
            item_r   = self.model.item(r, 2)
            item_g   = self.model.item(r, 3)
            item_b   = self.model.item(r, 4)

            # 빈 셀(None) 또는 빈 문자열("")이면 업데이트 중단
            if (item_g12 is None or item_g12.text().strip() == "" or
                item_r   is None or item_r.text().strip() == "" or
                item_g   is None or item_g.text().strip() == "" or
                item_b   is None or item_b.text().strip() == ""):
                return  # 아직 값이 완성되지 않은 상태 → 그래프 그리지 않음

            Gray12.append(float(item_g12.text()))
            Rvals.append(float(item_r.text()))
            Gvals.append(float(item_g.text()))
            Bvals.append(float(item_b.text()))

        # Keep gray12 sorted & unique
        Gray12 = np.array(Gray12, float)
        Rvals = np.array(Rvals, float)
        Gvals = np.array(Gvals, float)
        Bvals = np.array(Bvals, float)

        sort_idx = np.argsort(Gray12)
        Gray12 = Gray12[sort_idx]
        Rvals = Rvals[sort_idx]
        Gvals = Gvals[sort_idx]
        Bvals = Bvals[sort_idx]

        j_axis = np.arange(4096)
        R_full = np.interp(j_axis, Gray12, Rvals)
        G_full = np.interp(j_axis, Gray12, Gvals)
        B_full = np.interp(j_axis, Gray12, Bvals)

        self.curve_R.setData(j_axis, R_full)
        self.curve_G.setData(j_axis, G_full)
        self.curve_B.setData(j_axis, B_full)

    # ---------------------------------------------------------
    # 체크박스로 커브 on/off
    # ---------------------------------------------------------
    def update_curve_visibility(self):
        """
        체크박스 상태에 따라 각 곡선의 표시/숨김 제어
        """
        ck_R_H = getattr(self.ui, "ckBox_R_High", None)
        ck_G_H = getattr(self.ui, "ckBox_G_High", None)
        ck_B_H = getattr(self.ui, "ckBox_B_High", None)
        ck_R_L = getattr(self.ui, "ckBox_R_Low", None)
        ck_G_L = getattr(self.ui, "ckBox_G_Low", None)
        ck_B_L = getattr(self.ui, "ckBox_B_Low", None)

        if ck_R_H is not None:
            self.curve_R.setVisible(ck_R_H.isChecked())
        if ck_G_H is not None:
            self.curve_G.setVisible(ck_G_H.isChecked())
        if ck_B_H is not None:
            self.curve_B.setVisible(ck_B_H.isChecked())

        if ck_R_L is not None:
            self.curve_R_low.setVisible(ck_R_L.isChecked())
        if ck_G_L is not None:
            self.curve_G_low.setVisible(ck_G_L.isChecked())
        if ck_B_L is not None:
            self.curve_B_low.setVisible(ck_B_L.isChecked())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LUTEditor()
    win.show()
    sys.exit(app.exec())