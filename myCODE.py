#this is test file
# -*- coding: utf-8 -*-
import logging, json, pymysql
from datetime import datetime
import pandas as pd
from PySide2.QtCore import Slot, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 이미 보유한 스레드/차트
# from your_module import WriteVACdataThread, ReadVACdataThread, MeasureThread, GammaChart

# ─────────────────────────────────────────────────────────
# Matplotlib 캔버스 유틸: 레이아웃에 차트 1개 심기
# ─────────────────────────────────────────────────────────
def _mk_canvas_in_layout(self, target_layout):
    while target_layout.count():
        item = target_layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
    fig = Figure(figsize=(6,3), tight_layout=True)
    ax = fig.add_subplot(111)
    canvas = FigureCanvas(fig)
    target_layout.addWidget(canvas)
    return fig, ax, canvas

# ─────────────────────────────────────────────────────────
# VAC 상태 점검 및 OFF 전환 (당신의 check_VAC_status 사용)
# ─────────────────────────────────────────────────────────
def ensure_vac_off(self):
    """VAC 지원/활성 여부 확인 후, 활성(True)이면 luna로 Off 전환 시도."""
    st = self.check_VAC_status()
    if not st.get("supported", False):
        logging.info("VAC 미지원 모델: OFF 계측으로 진행합니다.")
        return True  # 어쨌든 진행

    if st.get("activated", False):
        logging.debug("VAC 활성 상태 감지 → VAC OFF 시도")
        # 질문에서 주신 'OFF' 커맨드 그대로 사용합니다.
        cmd = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff":true}\''
        self.send_command(self.ser_tv, 's')
        res = self.send_command(self.ser_tv, cmd)
        self.send_command(self.ser_tv, 'exit')
        logging.debug(f"VAC OFF 명령 응답: {res}")
        # 재확인
        st2 = self.check_VAC_status()
        if st2.get("activated", False):  # 여전히 True면 실패로 간주
            logging.warning("VAC OFF 실패로 보입니다. 그래도 측정은 진행합니다.")
        else:
            logging.info("VAC OFF 전환 성공.")
        return True
    else:
        logging.debug("이미 vac control off임. 측정 시작")
        return True

# ─────────────────────────────────────────────────────────
# DB 유틸: VAC OFF용(bypass) 1건 가져오기
# ─────────────────────────────────────────────────────────
def fetch_bypass_vac(self):
    self.send_command(self.ser_tv, 's')  # TV 세션 깨우기(당신 환경)
    db_conn = pymysql.connect(**self.conn_params)
    cursor = db_conn.cursor()
    query = f"SELECT `PK`,`VAC_Version`,`VAC_Data` FROM {self.DBName}.`W_VAC_Info` WHERE `Use_Flag`='N' AND `VAC_Version`='bypass' ORDER BY `PK` DESC LIMIT 1"
    cursor.execute(query)
    row = cursor.fetchone()
    cursor.close(); db_conn.close()
    self.send_command(self.ser_tv, 'exit')
    if not row:
        raise RuntimeError("bypass VAC을 DB에서 찾지 못했습니다.")
    pk, vac_version, vac_data = row
    try:
        vac_data = json.loads(vac_data) if isinstance(vac_data, str) else vac_data
    except Exception:
        pass
    return pk, vac_version, vac_data

# ─────────────────────────────────────────────────────────
# DB 유틸: 선택한 Panel Maker/Frame Rate에 매칭되는 VAC_Data 조회
# ─────────────────────────────────────────────────────────
def fetch_selected_vac_by_status(self):
    panel_maker = self.ui.vac_cmb_PanelMaker.currentText().strip()
    frame_rate  = self.ui.vac_cmb_FrameRate.currentText().strip()

    db_conn = pymysql.connect(**self.conn_params)
    cursor = db_conn.cursor()

    # W_VAC_Application_Status → VAC_Info_PK 찾기
    query1 = (
        f"SELECT `VAC_Info_PK` "
        f"FROM {self.DBName}.`W_VAC_Application_Status` "
        f"WHERE `Panel_Maker`=%s AND `Frame_Rate`=%s "
        f"ORDER BY `PK` DESC LIMIT 1"
    )
    cursor.execute(query1, (panel_maker, frame_rate))
    row = cursor.fetchone()
    if not row:
        cursor.close(); db_conn.close()
        raise RuntimeError("W_VAC_Application_Status에서 조건에 맞는 VAC_Info_PK를 찾지 못했습니다.")
    vac_info_pk = int(row[0])

    # W_VAC_Info → VAC_Data
    query2 = f"SELECT `VAC_Data` FROM {self.DBName}.`W_VAC_Info` WHERE `PK`=%s LIMIT 1"
    cursor.execute(query2, (vac_info_pk,))
    row2 = cursor.fetchone()
    cursor.close(); db_conn.close()
    if not row2:
        raise RuntimeError(f"W_VAC_Info PK={vac_info_pk} 의 VAC_Data를 찾지 못했습니다.")
    vac_data = row2[0]
    try:
        vac_data = json.loads(vac_data) if isinstance(vac_data, str) else vac_data
    except Exception:
        pass
    return vac_info_pk, vac_data

# ─────────────────────────────────────────────────────────
# LUT Plot / Table 업데이트
# ─────────────────────────────────────────────────────────
def plot_lut_to_box4(self, vac_data_dict):
    channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']
    lut = {ch: vac_data_dict.get(ch.replace("_","channel"), vac_data_dict.get(ch, [])) for ch in channels}
    df = pd.DataFrame(lut)

    fig, ax, canvas = _mk_canvas_in_layout(self, self.ui.vac_graph_rgbLUT_4)
    x = range(len(df))
    for ch in channels:
        ax.plot(x, df[ch], label=ch)
    ax.set_title("LUT (Applied)")
    ax.set_xlabel("Gray"); ax.set_ylabel("Value"); ax.legend(ncol=3)
    canvas.draw()

    # 테이블 업데이트
    tw = self.ui.vac_table_rbgLUT_4
    tw.setRowCount(len(df))
    tw.setColumnCount(len(channels))
    tw.setHorizontalHeaderLabels(channels)
    for r in range(len(df)):
        for c, ch in enumerate(channels):
            tw.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))

# ─────────────────────────────────────────────────────────
# 측정 공통: 결과를 차트/테이블에 반영
# ─────────────────────────────────────────────────────────
def append_gamma_point(self, role, pattern, gray, lv):
    """self.ui.vac_chart_gamma_3: 위(main)/아래(sub)에 점 추가"""
    line = self.lines_gamma[role][pattern]
    xs = list(line.get_xdata()); ys = list(line.get_ydata())
    xs.append(gray); ys.append(float(lv))
    line.set_data(xs, ys)
    self.gamma_chart.autoscale(); self.gamma_chart.draw()

def append_colorshift_point(self, role, x_xy, y_xy):
    """self.ui.vac_chart_colorShift_2: main/sub 함께 표시 (점만 추가)"""
    line = self.lines_cs[role]
    xs = list(line.get_xdata()); ys = list(line.get_ydata())
    xs.append(float(x_xy)); ys.append(float(y_xy))
    line.set_data(xs, ys)
    self.cs_canvas.draw()

# ─────────────────────────────────────────────────────────
# VAC OFF 측정 (결과: OFF baseline 저장)
# ─────────────────────────────────────────────────────────
def start_measurement_off(self):
    # 멀티축 감마 차트 준비(위=main, 아래=sub)
    self.gamma_chart = GammaChart(target_widget=self.ui.vac_chart_gamma_3, multi_axes=True, num_axes=2)
    self.lines_gamma = {
        'main': {p: self.gamma_chart.add_series(axis_index=0, label=f"OFF-main-{p}") for p in ['white','red','green','blue']},
        'sub' : {p: self.gamma_chart.add_series(axis_index=1, label=f"OFF-sub-{p}")  for p in ['white','red','green','blue']},
    }

    # 컬러쉬프트 차트 준비(둘이 같은 축)
    fig, ax, canvas = _mk_canvas_in_layout(self, self.ui.vac_chart_colorShift_2)
    ax.set_title("Color Shift (xy)"); ax.set_xlabel("x"); ax.set_ylabel("y")
    main_line, = ax.plot([], [], marker='o', linestyle='None', label='OFF-main')
    sub_line,  = ax.plot([], [], marker='^', linestyle='None', label='OFF-sub')
    ax.legend(); canvas.draw()
    self.cs_ax, self.cs_canvas = ax, canvas
    self.lines_cs = {'main': main_line, 'sub': sub_line}

    # OFF 결과 저장용 버퍼
    self.off_records = {
        'gamma': {'main': {}, 'sub': {}},   # e.g., {'white': {gray: (lv, cx, cy), ...}}
        'colorshift': {'main': [], 'sub': []}  # list of (x, y)
    }

    # 루프 상태
    self._meas_patterns = ['white','red','green','blue']
    self._meas_gray = list(range(256))
    self._pi = 0; self._gi = 0

    # 시작
    self._run_one_step_off()

def _run_one_step_off(self):
    if self._pi >= len(self._meas_patterns):
        # 색좌표 측정 단계(패치들) — 이미 op.colorshift_patterns를 쓰신다면 그대로
        self._cs_index = 0
        self._run_color_shift_off()
        return
    if self._gi >= len(self._meas_gray):
        self._gi = 0; self._pi += 1
        QTimer.singleShot(50, self._run_one_step_off); return

    pattern = self._meas_patterns[self._pi]
    gray = self._meas_gray[self._gi]
    # 패턴 RGB 출력
    if pattern == 'white': rgb = f"{gray},{gray},{gray}"
    elif pattern == 'red': rgb = f"{gray},0,0"
    elif pattern == 'green': rgb = f"0,{gray},0"
    else: rgb = f"0,0,{gray}"
    self.changeColor(rgb)

    # settle
    delay_ms = 3000 if (self._gi == 0 and self._pi == 0) else 400

    def trigger():
        buf = {}
        def handle(role, result):
            buf[role] = result
            if 'main' in buf and 'sub' in buf:
                # result = (x, y, lv, cct, duv)
                for role in ('main','sub'):
                    x, y, lv, cct, duv = buf[role]
                    # 차트
                    append_gamma_point(self, role, pattern, gray, lv)
                    # 표(OFF: main 테이블 1~3열 → 0,1,2)
                    if role == 'main':
                        row = self.ui.vac_table_measure_results_main_2.rowCount()
                        self.ui.vac_table_measure_results_main_2.insertRow(row)
                        self.ui.vac_table_measure_results_main_2.setItem(row, 0, QTableWidgetItem(str(lv)))
                        self.ui.vac_table_measure_results_main_2.setItem(row, 1, QTableWidgetItem(str(x)))  # Cx
                        self.ui.vac_table_measure_results_main_2.setItem(row, 2, QTableWidgetItem(str(y)))  # Cy
                    # 버퍼
                    self.off_records['gamma'][role].setdefault(pattern, {})[gray] = (float(lv), float(x), float(y))
                # 다음 gray
                self._gi += 1
                QTimer.singleShot(5, self._run_one_step_off)

        if self.main_instrument_cls:
            t1 = MeasureThread(self.main_instrument_cls, 'main')
            t1.measure_completed.connect(lambda r: handle('main', r)); t1.start()
        if self.sub_instrument_cls:
            t2 = MeasureThread(self.sub_instrument_cls, 'sub')
            t2.measure_completed.connect(lambda r: handle('sub', r)); t2.start()

    QTimer.singleShot(delay_ms, trigger)

def _run_color_shift_off(self):
    # 당신 프로젝트에서 쓰는 컬러쉬프트 패턴 시퀀스 사용
    seq = self.colorshift_patterns  # e.g., [('Red',255,0,0), ...]
    if self._cs_index >= len(seq):
        logging.info("[OFF] 측정 완료 → VAC 적용/검증 단계로 이동")
        self.apply_selected_vac_and_measure()
        return
    name, r, g, b = seq[self._cs_index]
    self.changeColor(f"{r},{g},{b}")

    def trigger():
        buf = {}
        def handle(role, result):
            buf[role] = result
            if 'main' in buf and 'sub' in buf:
                (x1, y1, lv1, cct1, duv1) = buf['main']
                (x2, y2, lv2, cct2, duv2) = buf['sub']
                append_colorshift_point(self, 'main', x1, y1)
                append_colorshift_point(self, 'sub',  x2, y2)
                self.off_records['colorshift']['main'].append((float(x1), float(y1)))
                self.off_records['colorshift']['sub'].append((float(x2), float(y2)))
                self._cs_index += 1
                QTimer.singleShot(50, self._run_color_shift_off)

        if self.main_instrument_cls:
            t1 = MeasureThread(self.main_instrument_cls, 'main')
            t1.measure_completed.connect(lambda r: handle('main', r)); t1.start()
        if self.sub_instrument_cls:
            t2 = MeasureThread(self.sub_instrument_cls, 'sub')
            t2.measure_completed.connect(lambda r: handle('sub', r)); t2.start()

    QTimer.singleShot(600, trigger)

# ─────────────────────────────────────────────────────────
# VAC 적용 → Read-back → LUT Plot/Table → ON 측정
# ─────────────────────────────────────────────────────────
def apply_selected_vac_and_measure(self):
    try:
        vac_info_pk, vac_data = fetch_selected_vac_by_status(self)
        logging.info(f"[ON] 적용 VAC_Info_PK={vac_info_pk}")

        # 쓰기
        self._w_thread = WriteVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName, vacdata_loaded=vac_data)
        self._w_thread.write_finished.connect(lambda ok, msg: self._on_write_done_on(ok, msg, vac_info_pk, vac_data))
        self._w_thread.start()
    except Exception as e:
        logging.exception(f"VAC 적용 실패: {e}")

def _on_write_done_on(self, ok, msg, vac_info_pk, vac_data):
    if not ok:
        logging.error(f"[ON] VAC write 실패: {msg}")
        return
    logging.info(f"[ON] VAC write 성공: {msg}")

    # Read-back
    self._r_thread = ReadVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName)
    self._r_thread.data_read.connect(lambda rd: self._on_read_done_on(rd, vac_info_pk))
    self._r_thread.error_occurred.connect(lambda err: logging.error(f"[ON] read 실패: {err}"))
    self._r_thread.start()

def _on_read_done_on(self, read_data, vac_info_pk):
    # LUT Plot + Table
    plot_lut_to_box4(self, read_data)

    # 감마 차트에 ON(위/아래) 라인 추가
    self.lines_gamma['main_on'] = {p: self.gamma_chart.add_series(axis_index=0, label=f"ON-main-{p}") for p in ['white','red','green','blue']}
    self.lines_gamma['sub_on']  = {p: self.gamma_chart.add_series(axis_index=1, label=f"ON-sub-{p}")  for p in ['white','red','green','blue']}

    # 색좌표 차트 ON 라인
    on_main_line, = self.cs_ax.plot([], [], marker='o', linestyle='None', label='ON-main')
    on_sub_line,  = self.cs_ax.plot([], [], marker='^', linestyle='None', label='ON-sub')
    self.cs_ax.legend(); self.cs_canvas.draw()
    self.lines_cs_on = {'main': on_main_line, 'sub': on_sub_line}

    # ON 결과 버퍼
    self.on_records = {'gamma': {'main': {}, 'sub': {}}, 'colorshift': {'main': [], 'sub': []}}

    # ON 측정 시작
    self._pi = 0; self._gi = 0
    self._run_one_step_on()

def _run_one_step_on(self):
    if self._pi >= len(self._meas_patterns):
        self._cs_index = 0
        self._run_color_shift_on()
        return
    if self._gi >= len(self._meas_gray):
        self._gi = 0; self._pi += 1
        QTimer.singleShot(50, self._run_one_step_on); return

    pattern = self._meas_patterns[self._pi]
    gray = self._meas_gray[self._gi]
    if pattern == 'white': rgb = f"{gray},{gray},{gray}"
    elif pattern == 'red': rgb = f"{gray},0,0"
    elif pattern == 'green': rgb = f"0,{gray},0"
    else: rgb = f"0,0,{gray}"
    self.changeColor(rgb)

    delay_ms = 400

    def trigger():
        buf = {}
        def handle(role, result):
            buf[role] = result
            if 'main' in buf and 'sub' in buf:
                for role in ('main','sub'):
                    x, y, lv, cct, duv = buf[role]
                    # 감마 차트의 ON 라인
                    line = self.lines_gamma['main_on' if role=='main' else 'sub_on'][pattern]
                    xs = list(line.get_xdata()); ys = list(line.get_ydata())
                    xs.append(gray); ys.append(float(lv))
                    line.set_data(xs, ys)
                    self.gamma_chart.autoscale(); self.gamma_chart.draw()

                    # 테이블(ON: 4~6열 → 3,4,5) & 오차(9~10열 → 8,9)
                    if role == 'main':
                        # 같은 gray 순서로 OFF에서 이미 한 줄씩 들어갔다고 가정 → 동일 row 사용
                        row = self._row_for_gray_off(gray)  # 아래에서 구현
                        self.ui.vac_table_measure_results_main_2.setItem(row, 3, QTableWidgetItem(str(lv)))
                        self.ui.vac_table_measure_results_main_2.setItem(row, 4, QTableWidgetItem(str(x)))
                        self.ui.vac_table_measure_results_main_2.setItem(row, 5, QTableWidgetItem(str(y)))
                        # 오차 = ON - OFF
                        off_lv, off_cx, off_cy = self.off_records['gamma']['main'][pattern].get(gray, (None,None,None))
                        if off_cx is not None and off_cy is not None:
                            dCx = float(x) - float(off_cx)
                            dCy = float(y) - float(off_cy)
                            self.ui.vac_table_measure_results_main_2.setItem(row, 8, QTableWidgetItem(f"{dCx:+.6f}"))
                            self.ui.vac_table_measure_results_main_2.setItem(row, 9, QTableWidgetItem(f"{dCy:+.6f}"))

                    # 버퍼 저장
                    self.on_records['gamma'][role].setdefault(pattern, {})[gray] = (float(lv), float(x), float(y))

                self._gi += 1
                QTimer.singleShot(5, self._run_one_step_on)

        if self.main_instrument_cls:
            t1 = MeasureThread(self.main_instrument_cls, 'main')
            t1.measure_completed.connect(lambda r: handle('main', r)); t1.start()
        if self.sub_instrument_cls:
            t2 = MeasureThread(self.sub_instrument_cls, 'sub')
            t2.measure_completed.connect(lambda r: handle('sub', r)); t2.start()

    QTimer.singleShot(delay_ms, trigger)

def _row_for_gray_off(self, gray):
    """OFF 측정 시 gray마다 1줄씩 append 했다는 가정 하에 row를 gray로 매핑."""
    # 가장 단순하게 gray == row 라고 두면 됩니다. (패턴별로 나누려면 별도 map 필요)
    # 패턴/멀티행 구조라면, OFF 측정 때 row 인덱스 맵을 만들어 두세요.
    return gray

def _run_color_shift_on(self):
    seq = self.colorshift_patterns
    if self._cs_index >= len(seq):
        logging.info("[ON] 측정 완료 → 오차 기반 보정 로직 호출")
        self.run_correction_logic()
        return

    name, r, g, b = seq[self._cs_index]
    self.changeColor(f"{r},{g},{b}")

    def trigger():
        buf = {}
        def handle(role, result):
            buf[role] = result
            if 'main' in buf and 'sub' in buf:
                (x1, y1, lv1, cct1, duv1) = buf['main']
                (x2, y2, lv2, cct2, duv2) = buf['sub']
                # ON color shift 추가
                line_m = self.lines_cs_on['main']; line_s = self.lines_cs_on['sub']
                for (line, x, y) in ((line_m,x1,y1),(line_s,x2,y2)):
                    xs = list(line.get_xdata()); ys = list(line.get_ydata())
                    xs.append(float(x)); ys.append(float(y))
                    line.set_data(xs, ys)
                self.cs_canvas.draw()
                self.on_records['colorshift']['main'].append((float(x1), float(y1)))
                self.on_records['colorshift']['sub'].append((float(x2), float(y2)))
                self._cs_index += 1
                QTimer.singleShot(50, self._run_color_shift_on)

        if self.main_instrument_cls:
            t1 = MeasureThread(self.main_instrument_cls, 'main')
            t1.measure_completed.connect(lambda r: handle('main', r)); t1.start()
        if self.sub_instrument_cls:
            t2 = MeasureThread(self.sub_instrument_cls, 'sub')
            t2.measure_completed.connect(lambda r: handle('sub', r)); t2.start()

    QTimer.singleShot(600, trigger)

# ─────────────────────────────────────────────────────────
# 보정 로직(자코비안 기반)을 여기서 호출
# ─────────────────────────────────────────────────────────
def run_correction_logic(self):
    """
    self.off_records / self.on_records 에서 ΔCx, ΔCy, ΔGamma 등을 꺼내
    'High만 조정' 자코비안 보정 루프를 호출하세요.
    (여기서는 훅만 두고, 당신의 보정 함수로 연결)
    """
    try:
        # 예) self.apply_high_update_via_jacobian(self.off_records, self.on_records)
        logging.info("보정 로직 호출 지점 (자코비안 업데이트 함수로 연결)")
    except Exception as e:
        logging.exception(f"보정 로직 실패: {e}")

# ─────────────────────────────────────────────────────────
# 메인 버튼 핸들러
# ─────────────────────────────────────────────────────────
@Slot()
def vac_btn_startOptimization_clicked(self):
    try:
        # 1) VAC OFF 보장
        if not ensure_vac_off(self):
            logging.error("VAC OFF 보장 실패")
            return

        # 2) bypass VAC을 실제로 TV에 한 번 써서(검증/plot) 시작하고 싶다면 주석 해제
        # pk, ver, vac_data = fetch_bypass_vac(self)
        # self._w0 = WriteVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName, vacdata_loaded=vac_data)
        # self._w0.write_finished.connect(lambda ok, msg: logging.info(f"bypass write: {ok}, {msg}"))
        # self._w0.start()

        # 3) VAC OFF 측정 시작
        start_measurement_off(self)

    except Exception as e:
        logging.exception(f"최적화 시작 실패: {e}")