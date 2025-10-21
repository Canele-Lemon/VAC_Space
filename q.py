class CIE1976ChromaticityDiagram:
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
                logging.warning(f"[CIE1976] 배경 이미지 없음: {image_path}")
        except Exception as e:
            logging.warning(f"[CIE1976] 배경 이미지 로드 실패: {e}")

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
            logging.warning(f"[CIE1976] BT.709 플롯 실패: {e}")

        try:
            DCI_u, DCI_v = cf.convert2DlistToPlot(op.DCI_uvprime)
            self.ax.plot(DCI_u, DCI_v, color='black', linestyle='-', linewidth=0.8, label="DCI")
        except Exception as e:
            logging.warning(f"[CIE1976] DCI 플롯 실패: {e}")

        try:
            CIE1976_u = [item[1] for item in op.CIE1976_uvprime]
            CIE1976_v = [item[2] for item in op.CIE1976_uvprime]
            self.ax.plot(CIE1976_u, CIE1976_v, color='black', linestyle='-', linewidth=0.3)
        except Exception as e:
            logging.warning(f"[CIE1976] CIE1976 곡선 플롯 실패: {e}")

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
                logging.warning(f"[CIE1976] Unknown series key: {key}")
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
            logging.exception(e)

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
                
class LUTChartVAC:
    """
    TV LUT(12-bit)를 매번 reset_and_plot()으로 새로 그립니다.
    CIE1976 차트와 동일한 cs.* 서식 적용.
    """
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
                logging.warning(f"[LUTChartVAC] {k} length invalid: {ys.size}")
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