from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import modules.chart_style as cs
import src.utils.common_functions as cf
import modules.optical_parameters as op
import numpy as np
import os

class CIE1976Chart:
    """
    - state: 'OFF' (레퍼런스, 빨강) / 'ON' (최적화/보정, 초록)
    - role:  'main' (0°) / 'sub' (60°)  → 마커: main=o, sub=s (hollow)
    - 배경 이미지/BT.709/DCI/CIE1976 등온선은 항상 표시
    - reset_on(): 'ON' 시리즈만 리셋
    - add_point(state, role, u_p, v_p): 한 점 추가 (각 세션/그레이별 누적)
    """
    def __init__(self, target_widget, title="Color Shift",
                 left_margin=0.10, right_margin=0.95, top_margin=0.95, bottom_margin=0.10):

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # ── 배경 이미지 ──
        try:
            image_path = cf.get_normalized_path(
                __file__, '..','..','..', 'resources/images/pictures', 'cie1976 (2).png'
            )
            if os.path.exists(image_path):
                img = plt.imread(image_path, format='png')
                self.ax.imshow(img, extent=[0, 0.70, 0, 0.60])
        except Exception as e:
            print(f"[CIE1976] BG load fail: {e}")

        # ── 서식(기존과 동일) ──
        cs.MatFormat_ChartArea(self.fig, left=left_margin, right=right_margin,
                               top=top_margin, bottom=bottom_margin)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title=title, color='#595959')
        cs.MatFormat_AxisTitle(self.ax, axis_title="u`", axis='x')
        cs.MatFormat_AxisTitle(self.ax, axis_title="v`", axis='y')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.7, tick_interval=0.1, axis='x')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.6, tick_interval=0.1, axis='y')
        cs.MatFormat_Gridline(self.ax, linestyle='--')

        # ── 레퍼런스 경계 ──
        try:
            BT709_u, BT709_v = cf.convert2DlistToPlot(op.BT709_uvprime)
            self.ax.plot(BT709_u, BT709_v, color='black', linestyle='--', linewidth=0.8, label="BT.709")
        except Exception as e:
            print(f"[CIE1976] BT.709 plot fail: {e}")
        try:
            DCI_u, DCI_v = cf.convert2DlistToPlot(op.DCI_uvprime)
            self.ax.plot(DCI_u, DCI_v, color='black', linestyle='-', linewidth=0.8, label="DCI")
        except Exception as e:
            print(f"[CIE1976] DCI plot fail: {e}")
        try:
            CIE1976_u = [r[1] for r in op.CIE1976_uvprime]
            CIE1976_v = [r[2] for r in op.CIE1976_uvprime]
            self.ax.plot(CIE1976_u, CIE1976_v, color='black', linestyle='-', linewidth=0.3)  # label=None
        except Exception as e:
            print(f"[CIE1976] iso plot fail: {e}")

        # ── 데이터 시리즈: (state, role) ──
        # 색: OFF=red, ON=green / 마커: main='o', sub='s' / hollow
        self._lines = {
            ('OFF','main'): self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='red')[0],
            ('OFF','sub') : self.ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='red')[0],
            ('ON','main') : self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='green')[0],
            ('ON','sub')  : self.ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='green')[0],
        }
        self._data  = {k: {'u':[], 'v':[]} for k in self._lines.keys()}
        self._update_legend()
        self.canvas.draw_idle()

    # ── public API ──
    def reset_on(self):
        """ON(보정/적용) 시리즈만 초기화; OFF(레퍼런스)는 유지."""
        for key in (('ON','main'), ('ON','sub')):
            self._data[key]['u'].clear()
            self._data[key]['v'].clear()
            self._lines[key].set_data([], [])
        self._update_legend()
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, u_p: float, v_p: float):
        """
        사용처 예:
          self.vac_optimization_cie1976_chart.add_point(
              state=('OFF' or 'ON'), role=('main' or 'sub'), u_p=..., v_p=...
          )
        """
        key = (state, role)
        if key not in self._lines:
            print(f"[CIE1976] invalid key: {key}")
            return
        self._data[key]['u'].append(float(u_p))
        self._data[key]['v'].append(float(v_p))
        self._lines[key].set_data(self._data[key]['u'], self._data[key]['v'])
        self._update_legend()
        self.canvas.draw_idle()

    # ── internals ──
    def _update_legend(self):
        handles, labels = [], []
        # 기준선
        for ln in self.ax.lines:
            if ln.get_label() in ("BT.709", "DCI"):
                handles.append(ln); labels.append(ln.get_label())
        # 시리즈 (데이터 있는 것만)
        label_map = {
            ('OFF','main'): "OFF main (0°)",
            ('OFF','sub') : "OFF sub (60°)",
            ('ON','main') : "ON main (0°)",
            ('ON','sub')  : "ON sub (60°)",
        }
        for key, ln in self._lines.items():
            if ln.get_xdata() and ln.get_ydata():
                ln.set_label(label_map[key])
                handles.append(ln); labels.append(label_map[key])

        if handles:
            self.ax.legend(handles, labels, fontsize=9, loc='lower right')
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()