# XYChart 안에 메서드 추가
def set_series(self, key, x_list, y_list, *, marker=None, linestyle='-', label=None, axis_index=0):
    """한 번에 x/y 전체를 세팅 (streaming 말고 배치 갱신)"""
    if key not in self.lines:
        # 미리 add_line 되어있지 않으면 생성
        axis = self.axes[axis_index]
        line, = axis.plot([], [], linestyle=linestyle, marker=marker, label=label or key)
        self.lines[key] = line
        self.data[key] = {'x': [], 'y': []}
    self.data[key]['x'] = list(x_list)
    self.data[key]['y'] = list(y_list)
    ln = self.lines[key]
    if marker is not None: ln.set_marker(marker)
    ln.set_linestyle(linestyle)
    if label is not None: ln.set_label(label)
    ln.set_data(self.data[key]['x'], self.data[key]['y'])
    ax = ln.axes
    ax.relim(); ax.autoscale_view()
    self.update_legend()
    self.canvas.draw()

def clear_series(self, key=None):
    """특정 시리즈만(또는 전체) 클리어"""
    if key is None:
        for k in list(self.lines.keys()):
            self.clear_series(k)
        return
    if key in self.lines:
        self.data[key]['x'].clear()
        self.data[key]['y'].clear()
        self.lines[key].set_data([], [])
        self.update_legend()
        self.canvas.draw()
        
# 새 클래스로 추가 (기존 BarChart는 그대로 두셔도 됩니다)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtWidgets import QSizePolicy
import modules.chart_style as cs

class GroupedBarChart:
    def __init__(self, target_widget, title='Grouped Bars',
                 x_labels=None, y_label='Value',
                 y_range=(0, 0.08), y_tick=0.02,
                 series_labels=('VAC OFF','VAC ON'),
                 spec_line=None):
        self.x_labels = x_labels or []
        self.y_label = y_label
        self.y_range = y_range
        self.y_tick = y_tick
        self.series_labels = series_labels
        self.spec_line = spec_line

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        cs.MatFormat_ChartArea(self.fig, left=0.12, right=0.96, top=0.9, bottom=0.18)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title)
        cs.MatFormat_AxisTitle(self.ax, self.y_label, axis='y')
        cs.MatFormat_Axis(self.ax, self.y_range[0], self.y_range[1], self.y_tick, axis='y')
        self.ax.set_xticks(np.arange(len(self.x_labels)))
        self.ax.set_xticklabels(self.x_labels, fontsize=9, color='#595959', rotation=0)
        self.ax.tick_params(axis='x', length=0)
        self.ax.legend(fontsize=8)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._bars = None
        self.canvas.draw()

    def update_grouped(self, data_off, data_on):
        """data_off/on: 길이=len(x_labels)"""
        self.ax.clear()
        # 스타일 재적용
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, self.ax.get_title())
        cs.MatFormat_AxisTitle(self.ax, self.y_label, axis='y')
        cs.MatFormat_Axis(self.ax, self.y_range[0], self.y_range[1], self.y_tick, axis='y')
        x = np.arange(len(self.x_labels))
        width = 0.38
        b1 = self.ax.bar(x - width/2, data_off, width, label=self.series_labels[0])
        b2 = self.ax.bar(x + width/2, data_on,  width, label=self.series_labels[1])
        if self.spec_line is not None:
            self.ax.axhline(self.spec_line, linestyle='--', linewidth=0.8, color='red')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(self.x_labels, fontsize=9, color='#595959')
        self.ax.legend(fontsize=8, loc='upper right')
        self.canvas.draw()
        
self.vac_optimization_chromaticity_chart = XYChart(
    target_widget=self.ui.vac_chart_chromaticityDiff,
    x_label='Gray Level', y_label='Cx/Cy',
    x_range=(0, 256), y_range=(0, 1),
    x_tick=64, y_tick=0.25,
    title=None, title_color='#595959',
    legend=True   # ← 변경
)

# 기존 colorshift 막대는 교체 (4패턴 * OFF/ON 묶음)
self.vac_optimization_colorshift_chart = GroupedBarChart(
    target_widget=self.ui.vac_chart_colorShift_3,
    title='Skin Color Shift',
    x_labels=['DarkSkin','LightSkin','Asian','Western'],
    y_label='Δu′v′',
    y_range=(0, 0.08), y_tick=0.02,
    series_labels=('VAC OFF','VAC ON'),
    spec_line=0.04
)
def _update_spec_views(self, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
    """
    요구하신 6개 위젯을 모두 갱신:
      1) vac_table_chromaticityDiff  (ΔCx/ΔCy/ΔGamma pass/total)
      2) vac_chart_chromaticityDiff  (Cx,Cy vs gray: OFF/ON)
      3) vac_table_gammaLinearity    (OFF/ON, 88~232 구간별 슬로프 평균)
      4) vac_chart_gammaLinearity    (8gray 블록 평균 슬로프 dot+line)
      5) vac_table_colorShift_3      (4 skin 패턴 Δu′v′, OFF/ON, 평균)
      6) vac_chart_colorShift_3      (Grouped bars)
    """
    import numpy as np

    # ===== 공통: white/main 시리즈 추출 =====
    def _extract_white(series_store):
        lv = np.full(256, np.nan, np.float64)
        cx = np.full(256, np.nan, np.float64)
        cy = np.full(256, np.nan, np.float64)
        for g in range(256):
            tup = series_store['gamma']['main']['white'].get(g, None)
            if tup:
                lv[g], cx[g], cy[g] = float(tup[0]), float(tup[1]), float(tup[2])
        return lv, cx, cy

    lv_off, cx_off, cy_off = _extract_white(off_store)
    lv_on , cx_on , cy_on  = _extract_white(on_store)

    # ===== 1) ChromaticityDiff 표: pass/total =====
    G_off = self._compute_gamma_series(lv_off)
    G_on  = self._compute_gamma_series(lv_on)
    dG  = np.abs(G_on - G_off)        # (256,)
    dCx = np.abs(cx_on - cx_off)
    dCy = np.abs(cy_on - cy_off)

    def _pass_total(arr, thr):
        mask = np.isfinite(arr)
        tot = int(np.sum(mask))
        ok  = int(np.sum((np.abs(arr[mask]) <= thr)))
        return ok, tot

    ok_cx, tot_cx = _pass_total(dCx, thr_c)
    ok_cy, tot_cy = _pass_total(dCy, thr_c)
    ok_g , tot_g  = _pass_total(dG , thr_gamma)

    # 표: (제목/헤더 제외) 2열×(2~4행) 채우기
    def _set_text(tbl, row, col, text):
        self._ensure_row_count(tbl, row)
        item = tbl.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            tbl.setItem(row, col, item)
        item.setText(text)

    tbl_ch = self.ui.vac_table_chromaticityDiff
    _set_text(tbl_ch, 1, 1, f"{ok_cx}/{tot_cx}")   # 2행,2열 ΔCx
    _set_text(tbl_ch, 2, 1, f"{ok_cy}/{tot_cy}")   # 3행,2열 ΔCy
    _set_text(tbl_ch, 3, 1, f"{ok_g}/{tot_g}")     # 4행,2열 ΔGamma

    # ===== 2) ChromaticityDiff 차트: Cx/Cy vs gray (OFF/ON) =====
    x = np.arange(256)
    self.vac_optimization_chromaticity_chart.set_series(
        "OFF_Cx", x, cx_off, marker=None, linestyle='-', label='OFF Cx'
    )
    self.vac_optimization_chromaticity_chart.set_series(
        "ON_Cx",  x, cx_on,  marker=None, linestyle='--', label='ON Cx'
    )
    self.vac_optimization_chromaticity_chart.set_series(
        "OFF_Cy", x, cy_off, marker=None, linestyle='-', label='OFF Cy'
    )
    self.vac_optimization_chromaticity_chart.set_series(
        "ON_Cy",  x, cy_on,  marker=None, linestyle='--', label='ON Cy'
    )

    # ===== 3) GammaLinearity 표: 88~232, 8gray 블록 평균 슬로프 =====
    def _segment_mean_slopes(lv_vec, g_start=88, g_end=232, step=8):
        # 1-step slope 정의(동일): 255*(lv[g+1]-lv[g])
        one_step = 255.0*(lv_vec[1:]-lv_vec[:-1])  # 0..254 개
        means = []
        for g in range(g_start, g_end, step):      # 88,96,...,224
            block = one_step[g:g+step]             # 8개
            m = np.nanmean(block) if np.isfinite(block).any() else np.nan
            means.append(m)
        return np.array(means, dtype=np.float64)   # 길이 = (232-88)/8 = 18개

    m_off = _segment_mean_slopes(lv_off)
    m_on  = _segment_mean_slopes(lv_on)
    avg_off = float(np.nanmean(m_off)) if np.isfinite(m_off).any() else float('nan')
    avg_on  = float(np.nanmean(m_on))  if np.isfinite(m_on).any()  else float('nan')

    tbl_gl = self.ui.vac_table_gammaLinearity
    _set_text(tbl_gl, 1, 1, f"{avg_off:.6f}")  # 2행,2열 OFF 평균 기울기
    _set_text(tbl_gl, 1, 2, f"{avg_on:.6f}")   # 2행,3열 ON  평균 기울기

    # ===== 4) GammaLinearity 차트: 블록 중심 x (= g+4), dot+line =====
    centers = np.arange(88, 232, 8) + 4    # 92,100,...,228
    self.vac_optimization_gammalinearity_chart.set_series(
        "OFF_slope8", centers, m_off, marker='o', linestyle='-', label='OFF slope(8)'
    )
    self.vac_optimization_gammalinearity_chart.set_series(
        "ON_slope8",  centers, m_on,  marker='o', linestyle='--', label='ON slope(8)'
    )

    # ===== 5) ColorShift(4종) 표 & 6) 묶음 막대 =====
    # store['colorshift'][role]에는 op.colorshift_patterns 순서대로 (x,y,u′,v′)가 append되어 있음
    # 우리가 필요로 하는 4패턴 인덱스 찾기
    want_names = ['Dark Skin','Light Skin','Asian','Western']   # op 리스트의 라벨과 동일하게
    name_to_idx = {name: i for i, (name, *_rgb) in enumerate(op.colorshift_patterns)}

    def _delta_uv_for_state(state_store):
        # main=정면(0°), sub=측면(60°) 가정
        arr = []
        for nm in want_names:
            idx = name_to_idx.get(nm, None)
            if idx is None: arr.append(np.nan); continue
            if idx >= len(state_store['colorshift']['main']) or idx >= len(state_store['colorshift']['sub']):
                arr.append(np.nan); continue
            _, _, u0, v0 = state_store['colorshift']['main'][idx]  # 정면
            _, _, u6, v6 = state_store['colorshift']['sub'][idx]   # 측면
            if not all(np.isfinite([u0,v0,u6,v6])):
                arr.append(np.nan); continue
            d = float(np.sqrt((u6-u0)**2 + (v6-v0)**2))
            arr.append(d)
        return np.array(arr, dtype=np.float64)  # [DarkSkin, LightSkin, Asian, Western]

    duv_off = _delta_uv_for_state(off_store)
    duv_on  = _delta_uv_for_state(on_store)
    mean_off = float(np.nanmean(duv_off)) if np.isfinite(duv_off).any() else float('nan')
    mean_on  = float(np.nanmean(duv_on))  if np.isfinite(duv_on).any()  else float('nan')

    # 표 채우기: 2열=OFF, 3열=ON / 2~5행=패턴 / 6행=평균
    tbl_cs = self.ui.vac_table_colorShift_3
    # OFF
    _set_text(tbl_cs, 1, 1, f"{duv_off[0]:.6f}")   # DarkSkin
    _set_text(tbl_cs, 2, 1, f"{duv_off[1]:.6f}")   # LightSkin
    _set_text(tbl_cs, 3, 1, f"{duv_off[2]:.6f}")   # Asian
    _set_text(tbl_cs, 4, 1, f"{duv_off[3]:.6f}")   # Western
    _set_text(tbl_cs, 5, 1, f"{mean_off:.6f}")     # 평균
    # ON
    _set_text(tbl_cs, 1, 2, f"{duv_on[0]:.6f}")
    _set_text(tbl_cs, 2, 2, f"{duv_on[1]:.6f}")
    _set_text(tbl_cs, 3, 2, f"{duv_on[2]:.6f}")
    _set_text(tbl_cs, 4, 2, f"{duv_on[3]:.6f}")
    _set_text(tbl_cs, 5, 2, f"{mean_on:.6f}")

    # 묶음 막대 차트 갱신
    self.vac_optimization_colorshift_chart.update_grouped(
        data_off=list(np.nan_to_num(duv_off, nan=0.0)),
        data_on =list(np.nan_to_num(duv_on,  nan=0.0))
    )
    
    
