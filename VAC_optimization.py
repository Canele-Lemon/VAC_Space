from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import modules.chart_style as cs

class GammaChart:
    """
    두 축(0° main / 60° sub), OFF는 참조로 유지, ON은 런마다 reset_on().
    CIE1976 차트와 동일한 서식(cs.MatFormat_*) 사용.
    """
    _PAT_COLORS = {'white':'gray', 'red':'red', 'green':'green', 'blue':'blue'}

    def __init__(self, target_widget, title='Gamma',
                 left=0.10, right=0.95, top=0.95, bottom=0.10,
                 x_tick=64, y_tick=None):
        # 두 개 축 (세로), x 공유
        self.fig, (self.ax_main, self.ax_sub) = plt.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)
        self.fig.subplots_adjust(hspace=0.0)

        # ── 공통 서식: CIE1976과 동일한 cs.* 사용 ──
        cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
        for i, (ax, atitle) in enumerate(((self.ax_main, 'Gamma'),
                                          (self.ax_sub,  'Gamma'))):
            cs.MatFormat_FigArea(ax)
            # (1) 제목 표시: 위쪽 축만
            if i == 0:
                cs.MatFormat_ChartTitle(ax, title=atitle, color='#595959')
            else:
                cs.MatFormat_ChartTitle(ax, title=None)  # 아래쪽 제목 제거

            # (2) x축 제목 및 눈금: 아래쪽 축만
            if i == 1:
                cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x')
                ax.tick_params(axis='x', which='both', labelbottom=True)  # 눈금 표시
            else:
                cs.MatFormat_AxisTitle(ax, axis_title='', axis='x')  # 숨김
                ax.tick_params(axis='x', which='both', labelbottom=False)  # 눈금 숨김

            # (3) y축 설정
            cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y')
            cs.MatFormat_Axis(ax, min_val=0, max_val=255, tick_interval=x_tick, axis='x')
            cs.MatFormat_Axis(ax, min_val=0, max_val=1, tick_interval=0.25, axis='y')
            cs.MatFormat_Gridline(ax, linestyle='--')

        # ── 시리즈: OFF/ON × main/sub × 패턴 ──
        self._lines = {}
        self._data  = {}
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            for pat, col in self._PAT_COLORS.items():
                # OFF(점선)
                k_off = ('OFF', role, pat)
                ln_off, = ax.plot([], [], linestyle='--', color=col, label=f'OFF {role} {pat}')
                self._lines[k_off] = ln_off; self._data[k_off] = {'x':[], 'y':[]}
                # ON(실선)
                k_on = ('ON', role, pat)
                ln_on, = ax.plot([], [], linestyle='-', color=col, label=f'ON {role} {pat}')
                self._lines[k_on] = ln_on; self._data[k_on] = {'x':[], 'y':[]}

        self._update_legends()
        self.canvas.draw_idle()

    def reset_on(self):
        """ON 시리즈만 리셋."""
        for key, ln in self._lines.items():
            if key[0] == 'ON':
                self._data[key]['x'].clear()
                self._data[key]['y'].clear()
                ln.set_data([], [])
        self._autoscale()
        self._update_legends()
        self.canvas.draw_idle()

    def add_point(self, *, state: str, role: str, pattern: str, gray: int, luminance: float):
        key = (state, role, pattern)
        if key not in self._lines:
            return
        self._data[key]['x'].append(int(gray))
        self._data[key]['y'].append(float(luminance))
        self._lines[key].set_data(self._data[key]['x'], self._data[key]['y'])
        self._autoscale(lazy_role=role)
        self.canvas.draw_idle()

    # ── 내부 유틸 ──
    def _autoscale(self, lazy_role=None):
        roles = [lazy_role] if lazy_role in ('main', 'sub') else ('main', 'sub')
        for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
            if role not in roles: continue
            ys = []
            for (state, r, pat), ln in self._lines.items():
                if r==role and ln.get_xdata() and ln.get_ydata():
                    ys.extend(ln.get_ydata())
            if ys:
                ymax = max(ys)
                ax.set_ylim(0, max(1e-6, ymax)*1.05)

    def _update_legends(self):
        for ax in (self.ax_main, self.ax_sub):
            handles, labels = [], []
            for (state, role, pat), ln in self._lines.items():
                if ln.axes is ax and ln.get_xdata() and ln.get_ydata():
                    handles.append(ln); labels.append(ln.get_label())
            if handles:
                ax.legend(handles, labels, fontsize=8, loc='upper left')
            else:
                leg = ax.get_legend()
                if leg: leg.remove()

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import modules.chart_style as cs

class LUTChart:
    def __init__(self, target_widget, title='TV LUT (12-bit)',
                 left=0.10, right=0.95, top=0.95, bottom=0.10,
                 x_tick=512, y_tick=512):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # ── CIE와 동일한 포맷 적용 ──
        cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title=title, color='#595959')
        cs.MatFormat_AxisTitle(self.ax, axis_title='Gray Level (12-bit)', axis='x')
        cs.MatFormat_AxisTitle(self.ax, axis_title='Input Level', axis='y')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=4095, tick_interval=x_tick, axis='x')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=4095, tick_interval=y_tick, axis='y')
        cs.MatFormat_Gridline(self.ax, linestyle='--')

        self._lines = {}
        self.canvas.draw_idle()

    def reset_and_plot(self, lut_dict: dict):
        """
        lut_dict = {
          "R_Low":[4096], "R_High":[4096],
          "G_Low":[4096], "G_High":[4096],
          "B_Low":[4096], "B_High":[4096],
        }
        """
        # 기존 라인 제거
        for ln in list(self._lines.values()):
            try: ln.remove()
            except Exception: pass
        self._lines.clear()

        xs = np.arange(4096)
        styles = {
            'R_Low':  dict(color='red',   ls='--', label='R Low'),
            'R_High': dict(color='red',   ls='-',  label='R High'),
            'G_Low':  dict(color='green', ls='--', label='G Low'),
            'G_High': dict(color='green', ls='-',  label='G High'),
            'B_Low':  dict(color='blue',  ls='--', label='B Low'),
            'B_High': dict(color='blue',  ls='-',  label='B High'),
        }
        ymax = 0.0
        for k, st in styles.items():
            ys = np.asarray(lut_dict.get(k, []), dtype=float).ravel()
            if ys.size != 4096:
                print(f"[LUTChartVAC] {k} length invalid: {ys.size}")
                continue
            ln, = self.ax.plot(xs, ys, **st)
            self._lines[k] = ln
            if ys.size:
                m = np.nanmax(ys)
                if np.isfinite(m): ymax = max(ymax, float(m))

        # 축/범례 갱신
        self.ax.set_xlim(0, 4095)
        self.ax.set_ylim(0, max(4095.0, ymax*1.05 if ymax>0 else 4095.0))
        if self._lines:
            self.ax.legend([ln for ln in self._lines.values()],
                           [ln.get_label() for ln in self._lines.values()],
                           fontsize=8, loc='upper left')
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()

        self.canvas.draw_idle()

import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import modules.chart_style as cs
import src.utils.common_functions as cf
import modules.optical_parameters as op

class CIE1976Chart:
    """
    - 배경 이미지: resources/images/pictures/cie1976 (2).png
    - 기준선: BT.709(점선), DCI(실선), CIE1976 등온선(가느다란 실선)
    - 데이터 포인트: data_1(OFF, 참조) = 빨강, data_2(ON) = 초록
        · 0deg: 원형(o, hollow)
        · 60deg: 사각형(s, hollow)
    - reset_on(): data_2_* 시리즈만 리셋 (참조 유지)
    - update(u_p, v_p, data_label, view_angle, vac_status): 기존 시그니처 유지
    """
    def __init__(self, target_widget, title="Color Shift",
                 left_margin=0.10, right_margin=0.95, top_margin=0.95, bottom_margin=0.10):
        import os
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        target_widget.addWidget(self.canvas)

        # ── 배경 이미지 로드 ──
        try:
            image_path = cf.get_normalized_path(
                __file__, '..','..','..', 'resources/images/pictures', 'cie1976 (2).png'
            )
            if os.path.exists(image_path):
                img = plt.imread(image_path, format='png')
                # 기존 initialize 코드와 동일한 extent
                self.ax.imshow(img, extent=[0, 0.70, 0, 0.60])
            else:
                print(f"[CIE1976] 배경 이미지 없음: {image_path}")
        except Exception as e:
            print(f"[CIE1976] 배경 이미지 로드 실패: {e}")

        # ── 스타일/눈금 (기존과 동일 포맷터 사용) ──
        cs.MatFormat_ChartArea(self.fig, left=left_margin, right=right_margin,
                               top=top_margin, bottom=bottom_margin)
        cs.MatFormat_FigArea(self.ax)
        cs.MatFormat_ChartTitle(self.ax, title=title, color='#595959')
        cs.MatFormat_AxisTitle(self.ax, axis_title='u`', axis='x')
        cs.MatFormat_AxisTitle(self.ax, axis_title='v`', axis='y')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.7, tick_interval=0.1, axis='x')
        cs.MatFormat_Axis(self.ax, min_val=0, max_val=0.6, tick_interval=0.1, axis='y')
        cs.MatFormat_Gridline(self.ax, linestyle='--')

        # ── 기준선(레퍼런스) ──
        try:
            BT709_u, BT709_v = cf.convert2DlistToPlot(op.BT709_uvprime)
            self.ax.plot(BT709_u, BT709_v, color='black', linestyle='--', linewidth=0.8, label="BT.709")
        except Exception as e:
            print(f"[CIE1976] BT.709 플롯 실패: {e}")

        try:
            DCI_u, DCI_v = cf.convert2DlistToPlot(op.DCI_uvprime)
            self.ax.plot(DCI_u, DCI_v, color='black', linestyle='-', linewidth=0.8, label="DCI")
        except Exception as e:
            print(f"[CIE1976] DCI 플롯 실패: {e}")

        try:
            CIE1976_u = [item[1] for item in op.CIE1976_uvprime]
            CIE1976_v = [item[2] for item in op.CIE1976_uvprime]
            self.ax.plot(CIE1976_u, CIE1976_v, color='black', linestyle='-', linewidth=0.3)
        except Exception as e:
            print(f"[CIE1976] CIE1976 곡선 플롯 실패: {e}")

        # ── 데이터 시리즈 (기존 initialize와 동일 스타일) ──
        # data_1: OFF(참조) 빨강, data_2: ON 초록
        self.lines = {
            'data_1_0deg': self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='red')[0],
            'data_1_60deg': self.ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='red')[0],
            'data_2_0deg': self.ax.plot([], [], 'o', markerfacecolor='none', markeredgecolor='green')[0],
            'data_2_60deg': self.ax.plot([], [], 's', markerfacecolor='none', markeredgecolor='green')[0],
        }
        self.data = {k: {'u': [], 'v': []} for k in self.lines.keys()}

        # 레전드: BT.709, DCI + (데이터 들어오면) 데이터 시리즈
        self._update_legend()
        self.canvas.draw()

    # ───────────────────────────
    # public API
    # ───────────────────────────
    def reset_on(self):
        """ON(data_2_*)만 리셋 → OFF(참조)는 유지"""
        for k in ('data_2_0deg', 'data_2_60deg'):
            self.data[k]['u'].clear()
            self.data[k]['v'].clear()
            self.lines[k].set_data([], [])
        self._update_legend()
        self.canvas.draw_idle()

    def update(self, u_p, v_p, data_label, view_angle, vac_status):
        """
        기존 호출부와 동일한 시그니처를 유지:
        - data_label: 'data_1'(OFF 참조) | 'data_2'(ON/보정)
        - view_angle: 0 | 60
        - vac_status: 레전드용 상태 텍스트 (예: 'VAC OFF (Ref.)', 'VAC ON', 'CORR#1' 등)
        """
        try:
            key = f'{data_label}_{int(view_angle)}deg'
            if key not in self.lines:
                print(f"[CIE1976] Unknown series key: {key}")
                return

            self.data[key]['u'].append(float(u_p))
            self.data[key]['v'].append(float(v_p))
            self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
            # 라벨 갱신(범례에서 보일 텍스트)
            friendly = '0°' if int(view_angle) == 0 else '60°'
            self.lines[key].set_label(f"{vac_status} - {data_label} {friendly}")

            self._update_legend()
            self.canvas.draw_idle()
        except Exception as e:
            print(e)

    # ───────────────────────────
    # internals
    # ───────────────────────────
    def _update_legend(self):
        handles, labels = [], []

        # 기준선(항상 표시)
        for ln in self.ax.lines:
            # CIE1976 곡선은 label None 처리했으니 제외됨
            if ln.get_label() in ("BT.709", "DCI"):
                handles.append(ln); labels.append(ln.get_label())

        # 데이터 시리즈(데이터 들어온 것만)
        for k in ('data_1_0deg','data_1_60deg','data_2_0deg','data_2_60deg'):
            ln = self.lines.get(k)
            if ln is not None and ln.get_xdata() and ln.get_ydata():
                handles.append(ln); labels.append(ln.get_label())

        if handles:
            self.ax.legend(handles, labels, fontsize=9, loc='lower right')
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()

class Widget_vacspace(QWidget):
    def __init__(self, parent=None):
        super(Widget_vacspace, self).__init__(parent)
        self.ui = Ui_vacspaceForm()
        self.ui.setupUi(self)

        self.ui.vac_btn_startOptimization.clicked.connect(self.start_VAC_optimization)
        self._vac_dict_cache = None
        
        self.vac_optimization_gamma_chart = GammaChart(self.ui.vac_chart_gamma_3)
        self.vac_optimization_cie1976_chart = CIE1976Chart(self.ui.vac_chart_colorShift_2)
        self.vac_optimization_lut_chart = LUTChart(target_widget=self.ui.vac_graph_rgbLUT_4)

        self.vac_optimization_chromaticity_chart = XYChart(
            target_widget=self.ui.vac_chart_chromaticityDiff,
            x_label='Gray Level',
            y_label='Cx/Cy',
            x_range=(0, 256),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title=None,
            title_color='#595959',
            legend=False
        )
        self.vac_optimization_gammalinearity_chart = XYChart(
            target_widget=self.ui.vac_chart_gammaLinearity,
            x_label='Gray Level',
            y_label='Slope',
            x_range=(0, 256),
            y_range=(0, 1),
            x_tick=64,
            y_tick=0.25,
            title=None,
            title_color='#595959',
            legend=False
        )
        self.vac_optimization_colorshift_chart = BarChart(
            target_widget=self.ui.vac_chart_colorShift_3,
            title='Skin Color Shift',
            x_labels=self.colorshift_x_labels,
            y_label='delta u`v`',
            spec_line=0.04
        )

        self.expanded = True
        self.expand_VAC_Optimization_Result()
        self.ui.vac_btn_JSONdownload.setEnabled(False)
        
        ###########################################################################################    
    
    def _load_jacobian_artifacts(self):
        """
        jacobian_Y0_high.pkl 파일을 불러와서 artifacts 딕셔너리로 반환
        """
        jac_path = cf.get_normalized_path(__file__, '.', 'models', 'jacobian2_Y0_high_INX_60.pkl')
        if not os.path.exists(jac_path):
            logging.error(f"[Jacobian] 파일을 찾을 수 없습니다: {jac_path}")
            raise FileNotFoundError(f"Jacobian model not found: {jac_path}")

        artifacts = joblib.load(jac_path)
        logging.info(f"[Jacobian] 모델 로드 완료: {jac_path}")
        print("======================= artifacts 구조 확인 =======================")
        logging.debug(f"Artifacts keys: {artifacts.keys()}")
        logging.debug(f"Components: {artifacts['components'].keys()}")
        return artifacts
    
    def _build_A_from_artifacts(self, artifacts, comp: str):
        """
        저장된 자코비안 pkl로부터 A 행렬 (ΔY ≈ A·ΔH) 복원
        """
        # def stack_basis_all_grays(knots: np.ndarray, L=256) -> np.ndarray:
        #     """
        #     모든 그레이(0..255)에 대한 φ(g) K차원 가중치 행렬 (L x K)
        #     """
        #     def linear_interp_weights(g: int, knots: np.ndarray) -> np.ndarray:
        #         """
        #         그레이 g(0..255)에 대해, K개 knot에 대한 선형보간 '모자(hat)' 가중치 벡터 φ(g) 반환.
        #         - 양 끝은 1개, 중간은 2개 노드만 비영(희소)
        #         """
        #         K = len(knots)
        #         w = np.zeros(K, dtype=np.float32)
        #         # 왼쪽/오른쪽 경계
        #         if g <= knots[0]:
        #             w[0] = 1.0
        #             return w
        #         if g >= knots[-1]:
        #             w[-1] = 1.0
        #             return w
        #         # 내부: 인접한 두 knot 사이
        #         i = np.searchsorted(knots, g) - 1
        #         g0, g1 = knots[i], knots[i+1]
        #         t = (g - g0) / max(1, (g1 - g0))
        #         w[i]   = 1.0 - t
        #         w[i+1] = t
        #         return w
            
            # rows = [linear_interp_weights(g, knots) for g in range(L)]
            # return np.vstack(rows).astype(np.float32)

        knots = np.asarray(artifacts["knots"], dtype=np.int32)
        comp_obj = artifacts["components"][comp]
        coef = np.asarray(comp_obj["coef"], dtype=np.float32)
        scale = np.asarray(comp_obj["standardizer"]["scale"], dtype=np.float32)

        s = comp_obj["feature_slices"]
        s_high_R = slice(s["high_R"][0], s["high_R"][1])
        s_high_G = slice(s["high_G"][0], s["high_G"][1])
        s_high_B = slice(s["high_B"][0], s["high_B"][1])

        beta_R = coef[s_high_R] / np.maximum(scale[s_high_R], 1e-12)
        beta_G = coef[s_high_G] / np.maximum(scale[s_high_G], 1e-12)
        beta_B = coef[s_high_B] / np.maximum(scale[s_high_B], 1e-12)

        Phi = self._stack_basis(knots, L=256)

        A_R = Phi * beta_R.reshape(1, -1)
        A_G = Phi * beta_G.reshape(1, -1)
        A_B = Phi * beta_B.reshape(1, -1)

        A = np.hstack([A_R, A_G, A_B]).astype(np.float32)
        logging.info(f"[Jacobian] {comp} A 행렬 shape: {A.shape}") # (256, 3K)
        return A
    
    def _set_vac_active(self, enable: bool) -> bool:
        try:
            logging.debug("현재 VAC 적용 상태를 확인합니다.")
            current_status = self._check_vac_status()
            current_active = bool(current_status.get("activated", False))

            if current_active == enable:
                logging.info(f"VAC already {'ON' if enable else 'OFF'} - skipping command.")
                return True

            self.send_command(self.ser_tv, 's')
            cmd = (
                "luna-send -n 1 -f "
                "luna://com.webos.service.panelcontroller/setVACActive "
                f"'{{\"OnOff\":{str(enable).lower()}}}'"
            )
            self.send_command(self.ser_tv, cmd)
            self.send_command(self.ser_tv, 'exit')
            time.sleep(0.5)
            st = self._check_vac_status()
            return bool(st.get("activated", False)) == enable
        
        except Exception as e:
            logging.error(f"VAC {'ON' if enable else 'OFF'} 전환 실패: {e}")
            return False
        
    def _check_vac_status(self):
        self.send_command(self.ser_tv, 's')
        getVACSupportstatus = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/getVACSupportStatus \'{"subscribe":true}\''
        VAC_support_status = self.send_command(self.ser_tv, getVACSupportstatus)
        VAC_support_status = self.extract_json_from_luna_send(VAC_support_status)
        self.send_command(self.ser_tv, 'exit')
        
        if not VAC_support_status:
            logging.warning("Failed to retrieve VAC support status from TV.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        if not VAC_support_status.get("isSupport", False):
            logging.info("VAC is not supported on this model.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        activated = VAC_support_status.get("isActivated", False)
        logging.info(f"VAC 적용 상태: {activated}")
                
        return {"supported": True, "activated": activated}
        
    def _dev_zero_lut_from_file(self):
        """원본 VAC JSON을 골라 6개 LUT 키만 0으로 덮어쓴 JSON을 임시파일로 저장하고 자동으로 엽니다."""
        # 1) 원본 JSON 선택
        fname, _ = QFileDialog.getOpenFileName(
            self, "원본 VAC JSON 선택", "", "JSON Files (*.json);;All Files (*)"
        )
        if not fname:
            return

        try:
            # 2) 순서 보존 로드
            with open(fname, "r", encoding="utf-8") as f:
                raw_txt = f.read()
            vac_dict = json.loads(raw_txt, object_pairs_hook=OrderedDict)

            # 3) LUT 6키를 모두 0으로 구성 (4096 포인트)
            zeros = np.zeros(4096, dtype=np.int32)
            zero_luts = {
                "RchannelLow":  zeros,
                "RchannelHigh": zeros,
                "GchannelLow":  zeros,
                "GchannelHigh": zeros,
                "BchannelLow":  zeros,
                "BchannelHigh": zeros,
            }

            vac_text = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=zero_luts)

            # 5) 임시파일로 저장
            fd, tmp_path = tempfile.mkstemp(prefix="VAC_zero_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(vac_text)

            # 6) startfile
            try:
                os.startfile(tmp_path)
            except Exception as e:
                logging.warning(f"임시파일 자동 열기 실패: {e}")

            QMessageBox.information(self, "완료", f"Zero-LUT JSON 임시파일 생성 및 열기 완료:\n{tmp_path}")

        except Exception as e:
            logging.exception(e)
            QMessageBox.critical(self, "오류", f"처리 중 오류: {e}")
        
    def build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict = None) -> str:
        """
        base_vac_dict : TV에서 읽은 원본 JSON(dict; 키 순서 유지 권장)
        new_lut_tvkeys: 교체할 LUT만 전달 시 병합 (TV 원 키명 그대로)
                        {"RchannelLow":[...4096], "RchannelHigh":[...], ...}
        return: TV에 바로 쓸 수 있는 탭 포맷 문자열
        """
        from collections import OrderedDict
        import numpy as np, json

        if not isinstance(base_vac_dict, (dict, OrderedDict)):
            raise ValueError("base_vac_dict must be dict/OrderedDict")

        od = OrderedDict(base_vac_dict)

        # 새 LUT 반영(형태/범위 보정)
        if new_lut_tvkeys:
            for k, v in new_lut_tvkeys.items():
                if k in od:
                    arr = np.asarray(v)
                    if arr.shape != (4096,):
                        raise ValueError(f"{k}: 4096 길이 필요 (현재 {arr.shape})")
                    od[k] = np.clip(arr.astype(int), 0, 4095).tolist()

        # -------------------------------
        # 포맷터
        # -------------------------------
        def _fmt_inline_list(lst):
            # [\t1,\t2,\t...\t]
            return "[\t" + ",\t".join(str(int(x)) for x in lst) + "\t]"

        def _fmt_list_of_lists(lst2d):
            """
            2D 리스트(예: DRV_valc_pattern_ctrl_1) 전용.
            마지막 닫힘은 ‘]\t\t]’ (쉼표 없음). 쉼표는 바깥 루프에서 1번만 붙임.
            """
            if not lst2d:
                return "[\t]"
            if not isinstance(lst2d[0], (list, tuple)):
                return _fmt_inline_list(lst2d)

            lines = []
            # 첫 행
            lines.append("[\t[\t" + ",\t".join(str(int(x)) for x in lst2d[0]) + "\t],")
            # 중간 행들
            for row in lst2d[1:-1]:
                lines.append("\t\t\t[\t" + ",\t".join(str(int(x)) for x in row) + "\t],")
            # 마지막 행(쉼표 없음) + 닫힘 괄호 정렬: “]\t\t]”
            last = "\t\t\t[\t" + ",\t".join(str(int(x)) for x in lst2d[-1]) + "\t]\t\t]"
            lines.append(last)
            return "\n".join(lines)

        def _fmt_flat_4096(lst4096):
            """
            4096 길이 LUT을 256x16으로 줄바꿈.
            마지막 줄은 ‘\t\t]’로 끝(쉼표 없음). 쉼표는 바깥에서 1번만.
            """
            a = np.asarray(lst4096, dtype=int)
            if a.size != 4096:
                raise ValueError(f"LUT 길이는 4096이어야 합니다. (현재 {a.size})")
            rows = a.reshape(256, 16)

            out = []
            # 첫 줄
            out.append("[\t" + ",\t".join(str(x) for x in rows[0]) + ",")
            # 중간 줄
            for r in rows[1:-1]:
                out.append("\t\t\t" + ",\t".join(str(x) for x in r) + ",")
            # 마지막 줄 (쉼표 X) + 닫힘
            out.append("\t\t\t" + ",\t".join(str(x) for x in rows[-1]) + "\t]")
            return "\n".join(out)

        lut_keys_4096 = {
            "RchannelLow","RchannelHigh",
            "GchannelLow","GchannelHigh",
            "BchannelLow","BchannelHigh",
        }

        # -------------------------------
        # 본문 생성
        # -------------------------------
        keys = list(od.keys())
        lines = ["{"]

        for i, k in enumerate(keys):
            v = od[k]
            is_last_key = (i == len(keys) - 1)
            trailing = "" if is_last_key else ","

            if isinstance(v, list):
                # 4096 LUT
                if k in lut_keys_4096 and len(v) == 4096 and not (v and isinstance(v[0], (list, tuple))):
                    body = _fmt_flat_4096(v)                       # 끝에 쉼표 없음
                    lines.append(f"\"{k}\"\t:\t{body}{trailing}")  # 쉼표는 여기서 1번만
                else:
                    # 일반 1D / 2D 리스트
                    if v and isinstance(v[0], (list, tuple)):
                        body = _fmt_list_of_lists(v)               # 끝에 쉼표 없음
                        lines.append(f"\"{k}\"\t:\t{body}{trailing}")
                    else:
                        body = _fmt_inline_list(v)                 # 끝에 쉼표 없음
                        lines.append(f"\"{k}\"\t:\t{body}{trailing}")

            elif isinstance(v, (int, float)):
                if k == "DRV_valc_hpf_ctrl_1":
                    lines.append(f"\"{k}\"\t:\t\t{int(v)}{trailing}")
                else:
                    lines.append(f"\"{k}\"\t:\t{int(v)}{trailing}")

            else:
                # 혹시 모를 기타 타입
                body = json.dumps(v, ensure_ascii=False)
                lines.append(f"\"{k}\"\t:\t{body}{trailing}")

        lines.append("}")
        return "\n".join(lines)
    
    def _run_off_baseline_then_on(self):
        profile_off = SessionProfile(
            legend_text="VAC OFF (Ref.)",
            cie_label="data_1",
            table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
            ref_store=None
        )

        def _after_off(store_off):
            self._off_store = store_off
            
            if not self._set_vac_active(True):
                logging.warning("VAC ON 전환 실패 - VAC 최적화 종료")
                return
                
            # 3. DB에서 모델/주사율에 맞는 VAC Data 적용 → 읽기 → LUT 차트 갱신
            self._apply_vac_from_db_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off, 
            gray_levels=op.gray_levels_256, 
            # gamma_patterns=('white','red','green','blue'),
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_off
        )
        
    def _apply_vac_from_db_and_measure_on(self):
        """
        3-a) DB에서 Panel_Maker + Frame_Rate 조합인 VAC_Data 가져오기
        3-b) TV에 쓰기 → TV에서 읽기
            → LUT 차트 갱신(reset_and_plot)
            → ON 시리즈 리셋(reset_on)
            → ON 측정 세션 시작(start_viewing_angle_session)
        """
        # 3-a) DB에서 VAC JSON 로드
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 종료")
            return

        # TV 쓰기 완료 시 콜백
        def _after_write(ok, msg):
            logging.info(f"[VAC Write] {msg}")
            if not ok:
                logging.error("VAC Writing 실패 - 종료")
                return
            # 쓰기 성공 → TV에서 VAC 읽어오기
            self._read_vac_from_tv(_after_read)

        # TV에서 읽기 완료 시 콜백
        def _after_read(vac_dict):
            if not vac_dict:
                logging.error("VAC 데이터 읽기 실패 - 종료")
                return

            # 캐시 보관 (TV 원 키명 유지)
            self._vac_dict_cache = vac_dict

            # LUT 차트는 "받을 때마다 전체 리셋 후 재그림"
            # TV 키명을 표준 표시용으로 바꿔서 전달 (RchannelHigh -> R_High 등)
            lut_plot = {
                key.replace("channel", "_"): v
                for key, v in vac_dict.items()
                if "channel" in key
            }
            # 새 LUT로 전체 리셋 후 플로팅
            self.vac_optimization_lut_chart.reset_and_plot(lut_plot)

            # ── ON 세션 시작 전: ON 시리즈 전부 리셋 ──
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            # ON 세션 프로파일 (OFF를 참조로 Δ 계산)
            profile_on = SessionProfile(
                legend_text="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            # ON 세션 종료 후: 스펙 체크 → 미통과면 보정 1회차 진입
            def _after_on(store_on):
                self._on_store = store_on
                if self._check_spec_pass(self._off_store, self._on_store):
                    logging.info("✅ 스펙 통과 — 종료")
                    return
                # (D) 반복 보정 시작 (1회차)
                self._run_correction_iteration(iter_idx=1)

            # ── ON 측정 세션 시작 ──
            # 간소화된 API: gamma_lines 인자 제거
            self.start_viewing_angle_session(
                profile=profile_on,
                gray_levels=getattr(op, "gray_levels_256", list(range(256))),
                gamma_patterns=('white','red','green','blue'),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                cs_settle_ms=1000,
                on_done=_after_on
            )

        # 3-b) VAC_Data TV에 writing
        self._write_vac_to_tv(vac_data, on_finished=_after_write)
        
    def _run_correction_iteration(self, iter_idx, max_iters=2, lambda_ridge=1e-3):
        logging.info(f"[CORR] iteration {iter_idx} start")

        # 1) 현재 TV LUT (캐시) 확보
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.warning("[CORR] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
            return None
        vac_dict = self._vac_dict_cache # 표준 키 dict

        # 2) 4096→256 다운샘플 (High만 수정, Low 고정)
        #    원래 키 → 표준 LUT 키로 꺼내 계산
        vac_lut = {
            "R_Low":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "R_High": np.asarray(vac_dict["RchannelHigh"], dtype=np.float32),
            "G_Low":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "G_High": np.asarray(vac_dict["GchannelHigh"], dtype=np.float32),
            "B_Low":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "B_High": np.asarray(vac_dict["BchannelHigh"], dtype=np.float32),
        }
        high_256 = {ch: self._down4096_to_256(vac_lut[ch]) for ch in ['R_High','G_High','B_High']}
        # low_256  = {ch: self._down4096_to_256(vac_lut[ch]) for ch in ['R_Low','G_Low','B_Low']}

        # 3) Δ 목표(white/main 기준): OFF vs ON 차이를 256 길이로 구성
        #    Gamma: 1..254 유효, Cx/Cy: 0..255
        d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
        # d_targets: {"Gamma":(256,), "Cx":(256,), "Cy":(256,)}

        # 4) 결합 선형계: [wG*A_Gamma; wC*A_Cx; wC*A_Cy] Δh = - [wG*ΔGamma; wC*ΔCx; wC*ΔCy]
        wG, wC = 1.0, 1.0
        A_cat = np.vstack([wG*self.A_Gamma, wC*self.A_Cx, wC*self.A_Cy]).astype(np.float32)
        b_cat = -np.concatenate([wG*d_targets["Gamma"], wC*d_targets["Cx"], wC*d_targets["Cy"]]).astype(np.float32)

        # 유효치 마스크(특히 gamma의 NaN)
        mask = np.isfinite(b_cat)
        A_use = A_cat[mask, :]
        b_use = b_cat[mask]

        # 5) 리지 해(Δh) 구하기 (3K-dim: [Rknots, Gknots, Bknots])
        #    (A^T A + λI) Δh = A^T b
        ATA = A_use.T @ A_use
        rhs = A_use.T @ b_use
        ATA[np.diag_indices_from(ATA)] += lambda_ridge
        delta_h = np.linalg.solve(ATA, rhs).astype(np.float32)

        # 6) Δcurve = Phi * Δh_channel 로 256-포인트 보정곡선 만들고 High에 적용
        K    = len(self._jac_artifacts["knots"])
        dh_R = delta_h[0:K]; dh_G = delta_h[K:2*K]; dh_B = delta_h[2*K:3*K]
        Phi  = self._stack_basis(self._jac_artifacts["knots"])  # (256,K)
        corr_R = Phi @ dh_R; corr_G = Phi @ dh_G; corr_B = Phi @ dh_B

        high_256_new = {
            "R_High": (high_256["R_High"] + corr_R).astype(np.float32),
            "G_High": (high_256["G_High"] + corr_G).astype(np.float32),
            "B_High": (high_256["B_High"] + corr_B).astype(np.float32),
        }

        # 7) High 경계/단조/클램프 → 12bit 업샘플 & Low는 유지하여 "표준 dict 구성"
        for ch in high_256_new:
            self._enforce_monotone(high_256_new[ch])
            high_256_new[ch] = np.clip(high_256_new[ch], 0, 4095)

        new_lut_tvkeys = {
            "RchannelLow":  np.asarray(self._vac_dict_cache["RchannelLow"], dtype=np.float32),
            "GchannelLow":  np.asarray(self._vac_dict_cache["GchannelLow"], dtype=np.float32),
            "BchannelLow":  np.asarray(self._vac_dict_cache["BchannelLow"], dtype=np.float32),
            "RchannelHigh": self._up256_to_4096(high_256_new["R_High"]),
            "GchannelHigh": self._up256_to_4096(high_256_new["G_High"]),
            "BchannelHigh": self._up256_to_4096(high_256_new["B_High"]),
        }

        vac_write_json = self.build_vacparam_std_format(self._vac_dict_cache, new_lut_tvkeys)

        def _after_write(ok, msg):
            logging.info(f"[VAC Write] {msg}")
            if not ok:
                return
            # 쓰기 성공 → 재읽기
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            if not vac_dict_after:
                logging.error("보정 후 VAC 재읽기 실패")
                return
            # ✅ 여기서 캐시 갱신 (성공 케이스에만)
            self._vac_dict_cache = vac_dict_after
            # 차트용 변환 후 표시
            lut_dict_plot = {k.replace("channel","_"): v
                            for k, v in vac_dict_after.items() if "channel" in k}
            self._update_lut_chart_and_table(lut_dict_plot)
            # 다음 측정 세션 시작 등...

        # TV에 적용
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)
            
            
            
            
    def _check_spec_pass(self, off_store, on_store, thr_gamma=0.05, thr_c=0.003):
    # white/main만 기준
        def _extract_white(series_store):
            lv = np.zeros(256); cx = np.zeros(256); cy = np.zeros(256)
            for g in range(256):
                tup = series_store['gamma']['main']['white'].get(g, None)
                if tup: lv[g], cx[g], cy[g] = tup
                else:   lv[g]=np.nan; cx[g]=np.nan; cy[g]=np.nan
            return lv, cx, cy

        lv_ref, cx_ref, cy_ref = _extract_white(off_store)
        lv_on , cx_on , cy_on  = _extract_white(on_store)

        G_ref = self._compute_gamma_series(lv_ref)
        G_on  = self._compute_gamma_series(lv_on)

        dG  = np.abs(G_on - G_ref)
        dCx = np.abs(cx_on - cx_ref)
        dCy = np.abs(cy_on - cy_ref)

        max_dG  = np.nanmax(dG)
        max_dCx = np.nanmax(dCx)
        max_dCy = np.nanmax(dCy)

        logging.info(f"[SPEC] max|ΔGamma|={max_dG:.6f} (≤{thr_gamma}), max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c})")
        return (max_dG <= thr_gamma) and (max_dCx <= thr_c) and (max_dCy <= thr_c)

    def _build_delta_targets_from_stores(self, off_store, on_store):
        # Δ = (ON - OFF). white/main
        lv_ref, cx_ref, cy_ref = np.zeros(256), np.zeros(256), np.zeros(256)
        lv_on , cx_on , cy_on  = np.zeros(256), np.zeros(256), np.zeros(256)
        for g in range(256):
            tR = off_store['gamma']['main']['white'].get(g, None)
            tO = on_store['gamma']['main']['white'].get(g, None)
            if tR: lv_ref[g], cx_ref[g], cy_ref[g] = tR
            else:  lv_ref[g]=np.nan; cx_ref[g]=np.nan; cy_ref[g]=np.nan
            if tO: lv_on[g], cx_on[g], cy_on[g] = tO
            else:  lv_on[g]=np.nan; cx_on[g]=np.nan; cy_on[g]=np.nan

        G_ref = self._compute_gamma_series(lv_ref)
        G_on  = self._compute_gamma_series(lv_on)
        d = {
            "Gamma": (G_on - G_ref),
            "Cx":    (cx_on - cx_ref),
            "Cy":    (cy_on - cy_ref),
        }
        # NaN → 0 (선형계 마스킹에서도 걸러지니 안정성↑)
        for k in d:
            d[k] = np.nan_to_num(d[k], nan=0.0).astype(np.float32)
        return d
    
    def start_viewing_angle_session(self,
        profile: SessionProfile,
        gray_levels=None,
        gamma_patterns=('white','red','green','blue'),
        colorshift_patterns=None,
        first_gray_delay_ms=3000,
        cs_settle_ms=1000,
        on_done=None,
    ):
        if gray_levels is None:
            gray_levels = op.gray_levels_256
        if colorshift_patterns is None:
            colorshift_patterns = op.colorshift_patterns

        store = {
            'gamma': {'main': {p:{} for p in gamma_patterns}, 'sub': {p:{} for p in gamma_patterns}},
            'colorshift': {'main': [], 'sub': []}
        }

        self._sess = {
            'phase': 'gamma',
            'p_idx': 0,
            'g_idx': 0,
            'cs_idx': 0,
            'patterns': list(gamma_patterns),
            'gray_levels': list(gray_levels),
            'cs_patterns': colorshift_patterns,
            'store': store,
            'profile': profile,
            'first_gray_delay_ms': first_gray_delay_ms,
            'cs_settle_ms': cs_settle_ms,
            'on_done': on_done
        }
        self._session_step()

    def _session_step(self):
        s = self._sess
        if s['phase'] == 'gamma':
            if s['p_idx'] >= len(s['patterns']):
                s['phase'] = 'colorshift'
                s['cs_idx'] = 0
                QTimer.singleShot(60, lambda: self._session_step())
                return

            if s['g_idx'] >= len(s['gray_levels']):
                s['g_idx'] = 0
                s['p_idx'] += 1
                QTimer.singleShot(40, lambda: self._session_step())
                return

            pattern = s['patterns'][s['p_idx']]
            gray = s['gray_levels'][s['g_idx']]

            if pattern == 'white':
                rgb_value = f"{gray},{gray},{gray}"
            elif pattern == 'red':
                rgb_value = f"{gray},0,0"
            elif pattern == 'green':
                rgb_value = f"0,{gray},0"
            else:
                rgb_value = f"0,0,{gray}"
            self.changeColor(rgb_value)

            delay = s['first_gray_delay_ms'] if s['g_idx'] == 0 else 0
            QTimer.singleShot(delay, lambda p=pattern, g=gray: self._trigger_gamma_pair(p, g))

        elif s['phase'] == 'colorshift':
            if s['cs_idx'] >= len(s['cs_patterns']):
                s['phase'] = 'done'
                QTimer.singleShot(0, lambda: self._session_step())
                return

            pname, r, g, b = s['cs_patterns'][s['cs_idx']]
            self.changeColor(f"{r},{g},{b}")
            QTimer.singleShot(s['cs_settle_ms'], lambda pn=pname: self._trigger_colorshift_pair(pn))

        else:  # done
            self._finalize_session()

    def _trigger_gamma_pair(self, pattern, gray):
        s = self._sess
        s['_gamma'] = {}

        def handle(role, res):
            s['_gamma'][role] = res
            got_main = 'main' in s['_gamma']
            got_sub = ('sub') in s['_gamma'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_gamma_pair(pattern, gray, s['_gamma'])
                s['g_idx'] += 1
                QTimer.singleShot(30, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle)
            self.sub_measure_thread.start()

    def _consume_gamma_pair(self, pattern, gray, results):
        """
        results: {
        'main': (x, y, lv, cct, duv)  또는  None,
        'sub' : (x, y, lv, cct, duv)  또는  None
        }
        """
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        # 현재 세션이 OFF 레퍼런스인지, ON/보정 런인지 상태 문자열 결정
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        # 두 역할을 results 키로 직접 순회 (측정기 객체 비교 X)
        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                # 측정 실패/결측인 경우
                store['gamma'][role][pattern][gray] = (np.nan, np.nan, np.nan)
                continue

            x, y, lv, cct, duv = res
            # 스토어 업데이트 (white 테이블/감마 계산 등에 사용)
            store['gamma'][role][pattern][gray] = (float(lv), float(x), float(y))

            # ▶▶ 차트 업데이트 (간소화 API)
            # GammaChartVAC: add_point(state, role, pattern, gray, luminance)
            self.vac_optimization_gamma_chart.add_point(
                state=state,
                role=role,               # 'main' 또는 'sub'
                pattern=pattern,         # 'white'|'red'|'green'|'blue'
                gray=int(gray),
                luminance=float(lv)
            )

        # (아래 white/main 테이블 채우는 로직은 기존 그대로 유지)
        if pattern == 'white':
            # main 테이블
            lv_m, cx_m, cy_m = store['gamma']['main']['white'].get(gray, (np.nan, np.nan, np.nan))
            table_inst1 = self.ui.vac_table_opt_mes_results_main
            cols = profile.table_cols
            self._set_item(table_inst1, gray, cols['lv'], f"{lv_m:.6f}" if np.isfinite(lv_m) else "")
            self._set_item(table_inst1, gray, cols['cx'], f"{cx_m:.6f}" if np.isfinite(cx_m) else "")
            self._set_item(table_inst1, gray, cols['cy'], f"{cy_m:.6f}" if np.isfinite(cy_m) else "")

            # sub 테이블
            lv_s, cx_s, cy_s = store['gamma']['sub']['white'].get(gray, (np.nan, np.nan, np.nan))
            table_inst2 = self.ui.vac_table_opt_mes_results_sub
            self._set_item(table_inst2, gray, cols['lv'], f"{lv_s:.6f}" if np.isfinite(lv_s) else "")
            self._set_item(table_inst2, gray, cols['cx'], f"{cx_s:.6f}" if np.isfinite(cx_s) else "")
            self._set_item(table_inst2, gray, cols['cy'], f"{cy_s:.6f}" if np.isfinite(cy_s) else "")

            # ΔCx/ΔCy (ON 세션에서만; ref_store가 있을 때)
            if profile.ref_store is not None and 'd_cx' in cols and 'd_cy' in cols:
                ref_main = profile.ref_store['gamma']['main']['white'].get(gray, None)
                if ref_main is not None:
                    _, cx_r, cy_r = ref_main
                    if np.isfinite(cx_m): self._set_item(table_inst1, gray, cols['d_cx'], f"{(cx_m - cx_r):.6f}")
                    if np.isfinite(cy_m): self._set_item(table_inst1, gray, cols['d_cy'], f"{(cy_m - cy_r):.6f}")

                ref_sub = profile.ref_store['gamma']['sub']['white'].get(gray, None)
                if ref_sub is not None:
                    _, cx_r_s, cy_r_s = ref_sub
                    if np.isfinite(cx_s): self._set_item(table_inst2, gray, cols['d_cx'], f"{(cx_s - cx_r_s):.6f}")
                    if np.isfinite(cy_s): self._set_item(table_inst2, gray, cols['d_cy'], f"{(cy_s - cy_r_s):.6f}")

    def _trigger_colorshift_pair(self, patch_name):
        s = self._sess
        s['_cs'] = {}

        def handle(role, res):
            s['_cs'][role] = res
            got_main = 'main' in s['_cs']
            got_sub = ('sub') in s['_cs'] or (self.sub_instrument_cls is None)
            if got_main and got_sub:
                self._consume_colorshift_pair(patch_name, s['_cs'])
                s['cs_idx'] += 1
                QTimer.singleShot(80, lambda: self._session_step())

        if self.main_instrument_cls:
            self.main_measure_thread = MeasureThread(self.main_instrument_cls, 'main')
            self.main_measure_thread.measure_completed.connect(handle)
            self.main_measure_thread.start()

        if self.sub_instrument_cls:
            self.sub_measure_thread = MeasureThread(self.sub_instrument_cls, 'sub')
            self.sub_measure_thread.measure_completed.connect(handle)
            self.sub_measure_thread.start()

    def _consume_colorshift_pair(self, patch_name, results):
        """
        results: {
        'main': (x, y, lv, cct, duv)  또는  None,
        'sub' : (x, y, lv, cct, duv)  또는  None
        }
        """
        s = self._sess
        store = s['store']
        profile: SessionProfile = s['profile']

        # 현재 세션 상태
        state = 'OFF' if profile.legend_text.startswith('VAC OFF') else 'ON'

        for role in ('main', 'sub'):
            res = results.get(role, None)
            if res is None:
                store['colorshift'][role].append((np.nan, np.nan, np.nan, np.nan))
                continue

            x, y, lv, cct, duv = res

            # xy → u′v′ 변환
            u_p, v_p = cf.convert_xyz_to_uvprime(float(x), float(y))
            store['colorshift'][role].append((float(x), float(y), float(u_p), float(v_p)))

            # ▶▶ 차트 업데이트 (간소화 API)
            # CIE1976ChartVAC: add_point(state, role, u_p, v_p)
            self.vac_optimization_cie1976_chart.add_point(
                state=state,
                role=role,        # 'main' 또는 'sub'
                u_p=float(u_p),
                v_p=float(v_p)
            )

    def _finalize_session(self):
        s = self._sess
        profile: SessionProfile = s['profile']
        table = self.ui.vac_table_opt_mes_results_main
        cols = profile.table_cols

        # white/main 감마 계산
        lv_series = np.zeros(256, dtype=np.float64)
        for g in range(256):
            tup = s['store']['gamma']['main']['white'].get(g, None)
            lv_series[g] = float(tup[0]) if tup else np.nan
        gamma_vec = self._compute_gamma_series(lv_series)
        for g in range(256):
            if np.isfinite(gamma_vec[g]):
                self._set_item(table, g, cols['gamma'], f"{gamma_vec[g]:.6f}")

        # ΔGamma (ON/보정 시)
        if profile.ref_store is not None and 'd_gamma' in cols:
            ref_lv = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = profile.ref_store['gamma']['main']['white'].get(g, None)
                ref_lv[g] = float(tup[0]) if tup else np.nan
            ref_gamma = self._compute_gamma_series(ref_lv)
            dG = gamma_vec - ref_gamma
            for g in range(256):
                if np.isfinite(dG[g]):
                    self._set_item(table, g, cols['d_gamma'], f"{dG[g]:.6f}")

        if callable(s['on_done']):
            try:
                s['on_done'](s['store'])
            except Exception as e:
                logging.exception(e)
                
    def _ensure_row_count(self, table, row_idx):
        if table.rowCount() <= row_idx:
            table.setRowCount(row_idx + 1)

    def _set_item(self, table, row, col, value):
        self._ensure_row_count(table, row)
        table.setItem(row, col, QTableWidgetItem("" if value is None else str(value)))

    def _compute_gamma_series(self, lv_vec_256):
        lv = np.asarray(lv_vec_256, dtype=np.float64)
        gamma = np.full(256, np.nan, dtype=np.float64)
        lv0 = lv[0]
        denom = np.max(lv[1:] - lv0)
        if not np.isfinite(denom) or denom <= 0:
            return gamma
        nor = (lv - lv0) / denom
        gray = np.arange(256, dtype=np.float64)
        gray_norm = gray / 255.0
        valid = (gray >= 1) & (gray <= 254) & (nor > 0) & np.isfinite(nor)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma[valid] = np.log(nor[valid]) / np.log(gray_norm[valid])
        return gamma
    
    def _stack_basis(self, knots, L=256):
        knots = np.asarray(knots, dtype=np.int32)
        
        def _phi(g):
            # 선형 모자 함수
            K = len(knots)
            w = np.zeros(K, dtype=np.float32)
            if g <= knots[0]:
                w[0]=1.; return w
            if g >= knots[-1]:
                w[-1]=1.; return w
            i = np.searchsorted(knots, g) - 1
            g0, g1 = knots[i], knots[i+1]
            t = (g - g0) / max(1, (g1 - g0))
            w[i] = 1-t; w[i+1] = t
            return w
        return np.vstack([_phi(g) for g in range(L)])

    def _down4096_to_256(self, arr4096):
        arr4096 = np.asarray(arr4096, dtype=np.float32)
        idx = np.round(np.linspace(0, 4095, 256)).astype(int)
        return arr4096[idx]

    def _up256_to_4096(self, arr256):
        arr256 = np.asarray(arr256, dtype=np.float32)
        x_small = np.linspace(0, 1, 256)
        x_big   = np.linspace(0, 1, 4096)
        return np.interp(x_big, x_small, arr256).astype(np.float32)

    def _enforce_monotone(self, arr):
        # 제자리 누적 최대치
        for i in range(1, len(arr)):
            if arr[i] < arr[i-1]:
                arr[i] = arr[i-1]
        return arr
        
    def _fetch_vac_by_model(self, panel_maker, frame_rate):
        """
        DB: W_VAC_Application_Status에서 Panel_Maker/Frame_Rate 매칭 → VAC_Info_PK 얻고
            W_VAC_Info.PK=VAC_Info_PK → VAC_Data 읽어서 반환
        반환: (pk, vac_version, vac_data)  또는 (None, None, None)
        """
        try:
            db_conn= pymysql.connect(**config.conn_params)
            cursor = db_conn.cursor()

            cursor.execute("""
                SELECT `VAC_Info_PK`
                FROM `W_VAC_Application_Status` 
                WHERE Panel_Maker = %s AND Frame_Rate = %s
            """, (panel_maker, frame_rate))

            result = cursor.fetchone()

            if not result:
                logging.error("No VAC_Info_PK found for given Panel Maker/Frame Rate")
                return None, None, None
            
            vac_info_pk = result[0]          
            logging.debug(f"VAC_Info_PK = {vac_info_pk}")

            cursor.execute("""
                SELECT `VAC_Version`, `VAC_Data`
                FROM `W_VAC_Info`
                WHERE `PK` = %s
            """, (vac_info_pk,))

            vac_row = cursor.fetchone()

            if not vac_row:
                logging.error(f"No VAC information found for PK={vac_info_pk}")
                return None, None, None

            vac_version = vac_row[0]
            vac_data = vac_row[1]
            
            return vac_info_pk, vac_version, vac_data
        
        except Exception as e:
            logging.exception(e)
            return None, None, None
        
        finally:
            if db_conn:
                db_conn.close()

    def _write_vac_to_tv(self, vac_data, on_finished):
        t = WriteVACdataThread(parent=self, ser_tv=self.ser_tv,
                                vacdataName=self.vacdataName, vacdata_loaded=vac_data)
        t.write_finished.connect(lambda ok, msg: on_finished(ok, msg))
        t.start()

    def _read_vac_from_tv(self, on_finished):
        t = ReadVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName)
        t.data_read.connect(lambda data: on_finished(data))
        t.error_occurred.connect(lambda err: (logging.error(err), on_finished(None)))
        t.start()

    def _update_lut_chart_and_table(self, lut_dict):
        try:
            required = ["R_Low", "R_High", "G_Low", "G_High", "B_Low", "B_High"]
            for k in required:
                if k not in lut_dict:
                    logging.error(f"missing key: {k}")
                    return
                if len(lut_dict[k]) != 4096:
                    logging.error(f"invalid length for {k}: {len(lut_dict[k])} (expected 4096)")
                    return
                
            df = pd.DataFrame({
                "R_Low":  lut_dict["R_Low"],
                "R_High": lut_dict["R_High"],
                "G_Low":  lut_dict["G_Low"],
                "G_High": lut_dict["G_High"],
                "B_Low":  lut_dict["B_Low"],
                "B_High": lut_dict["B_High"],
            })
            self.update_rgbchannel_table(df, self.ui.vac_table_rbgLUT_4)
            
            chart = self.vac_optimization_lut_chart
            xs = np.arange(4096, dtype=int).tolist()
            series_meta = [
                ("R_Low",  "R Low",  "red",   "--"),
                ("R_High", "R High", "red",   "-"),
                ("G_Low",  "G Low",  "green", "--"),
                ("G_High", "G High", "green", "-"),
                ("B_Low",  "B Low",  "blue",  "--"),
                ("B_High", "B High", "blue",  "-"),
            ]
            
            for col, label, color, ls in series_meta:
                ys = df[col].astype(float).tolist()
                
                if label not in chart.lines:
                    chart.add_line(key=label, color=color, linestyle=ls, axis_index=0, label=label)
        
                chart.data[label]['x'] = xs
                chart.data[label]['y'] = ys
                
                line = chart.lines[label]
                line.set_data(chart.data[label]['x'], chart.data[label]['y'])
                
            for ax in chart.axes:
                ax.relim()
                ax.autoscale_view()
            chart.canvas.draw()
        
        except Exception as e:
            logging.exception(e)

    def start_VAC_optimization(self):
        
        """
        =============== 메인 엔트리: 버튼 이벤트 연결용 ===============
        전체 Flow:
        1) TV setting > VAC OFF → 측정(OFF baseline) + UI 업데이트
            - 

        2) TV setting > VAC OFF → DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 writing → 측정(ON 현재) + UI 업데이트

        3) 스펙 확인 → 통과면 종료
        
        4) 미통과면 자코비안 기반 보정(256기준) → 4096 보간 반영 → 예측모델 검증 → OK면 → TV 적용 → 재측정 → 스펙 재확인
        5) (필요 시 반복 2~3회만)
        """
        self.label_processing_step_1, self.movie_processing_step_1 = self.start_loading_animation(self.ui.vac_label_pixmap_step_1, 'processing.gif')
        try:
            # 0. 자코비안 로드
            artifacts = self._load_jacobian_artifacts()
            self._jac_artifacts = artifacts
            print("======================= A 행렬 shape 확인 =======================")
            self.A_Gamma = self._build_A_from_artifacts(artifacts, "Gamma")   # (256, 3K)
            self.A_Cx    = self._build_A_from_artifacts(artifacts, "Cx")
            self.A_Cy    = self._build_A_from_artifacts(artifacts, "Cy")

        except FileNotFoundError as e:
            logging.error(f"[VAC Optimization] Jacobian file not found: {e}")

        except KeyError as e:
            logging.error(f"[VAC Optimization] Missing key in artifacts: {e}")

        except Exception as e:
            logging.exception("[VAC Optimization] Unexpected error occurred")

        # 1. VAC OFF 보장 + 측정
        # 1.1 결과 저장용 버퍼 초기화 (OFF / ON 구분)
        self._off_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        self._on_store  = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}}, 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                        'colorshift': {'main': [], 'sub': []}}
        # 1.2 TV VAC OFF 하기
        if not self._set_vac_active(False):
            logging.error("VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
            
        # 1.3 OFF 측정 세션 시작
        self._run_off_baseline_then_on()
