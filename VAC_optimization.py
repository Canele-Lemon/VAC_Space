# -*- coding: utf-8 -*-
# PySide2 기반: 여러분의 메인 윈도우 클래스(예: MainWindow) 내부에 그대로 넣어 쓰면 됩니다.

from PySide2.QtCore import QObject, Signal, Slot, QTimer
from PySide2.QtWidgets import QTableWidgetItem
import logging, json, pymysql, numpy as np, pandas as pd

# =========================
# 1) 시작 버튼 핸들러
# =========================
def start_VAC_optimization(self):
    """
    버튼 연결: self.ui.vac_btn_startOptimization.clicked.connect(self.start_VAC_optimization)
    플로우:
      1) VAC OFF 보장 → OFF 측정
      2) DB에서 대상 LUT 조회/적용 → Read-back → ON 측정
      3) Diff 계산 & 표 업데이트 → 보정 로직 호출
    """
    try:
        # 1) VAC OFF 보장
        st = self.check_VAC_status()
        if st.get("activated", False):
            logging.debug("VAC 활성 상태 감지 → VAC OFF 시도")
            cmd_off = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff": false}\''
            self.send_command(self.ser_tv, 's')
            res = self.send_command(self.ser_tv, cmd_off)
            self.send_command(self.ser_tv, 'exit')
            logging.debug(f"VAC OFF 명령 응답: {res}")

            # 재확인
            st2 = self.check_VAC_status()
            if st2.get("activated", False):
                logging.warning("VAC OFF 실패로 보입니다. 그래도 진행합니다(OFF로 간주).")
            else:
                logging.info("VAC OFF 전환 성공.")
        else:
            logging.debug("이미 VAC control OFF 상태. OFF 측정 시작.")

        # 2) VAC OFF 상태 측정
        self._measure_for_mode(
            mode_label="OFF",
            on_finished=self._after_off_measured
        )

    except Exception as e:
        logging.exception(f"[start_VAC_optimization] 실패: {e}")


# =========================
# 2) OFF 측정 완료 → ON 단계 준비
# =========================
def _after_off_measured(self, off_result):
    """
    off_result: dict
      {
        "gamma_main": {("W",g): Lv/Cx/Cy ...},  # 구현에 따라 구조는 앱 내 표준으로
        "gamma_sub":  {...},
        "colorshift": {...},                    # main/sub 좌표
        "table_rows_main": [(Lv, Cx, Cy) ...], # 테이블 반영에도 사용 가능
      }
    """
    try:
        self._off_result = off_result  # 다음 단계에서 Diff 계산 때 사용

        # 3) DB에서 모델/주사율에 맞는 VAC LUT 조회 & TV 적용
        vac_pk, vac_data_json = self._fetch_target_vac_from_db()
        if not vac_data_json:
            logging.error("DB에서 대상 VAC LUT를 가져오지 못했습니다. 중단합니다.")
            return

        # TV Writing (스레드 사용)
        self._write_vac_to_tv(
            vac_data_json=vac_data_json,
            on_written=lambda ok, msg: self._after_tv_written(ok, msg, vac_pk)
        )

    except Exception as e:
        logging.exception(f"[_after_off_measured] 실패: {e}")


# =========================
# 3) TV에 LUT Writing → Readback 후 차트/테이블 갱신
# =========================
def _after_tv_written(self, ok, msg, vac_pk):
    if not ok:
        logging.warning(f"[VAC Write] 실패: {msg}")
        # 그래도 진행할지, 중단할지는 정책에 따라 결정
        return

    logging.info(f"[VAC Write] 성공: PK={vac_pk}, msg={msg}")

    # Read-back으로 실제 적용 LUT 확인
    self._read_vac_from_tv(on_read=self._after_tv_readback)


def _after_tv_readback(self, read_data_dict):
    """
    read_data_dict: {"R_Low":[...], "R_High":[...], ...} 형태라고 가정
    """
    try:
        # 그래프/테이블 업데이트
        rgb_df = pd.DataFrame(read_data_dict)
        self.update_rgbchannel_chart(
            rgb_df,
            self.graph['vac_laboratory']['data_acquisition_system']['input']['ax'],
            self.graph['vac_laboratory']['data_acquisition_system']['input']['canvas']
        )
        self.update_rgbchannel_table(rgb_df, self.ui.vac_table_rbgLUT_4)

        # 4) VAC ON으로 전환(기능 활성화)
        cmd_on = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff": true}\''
        self.send_command(self.ser_tv, 's')
        res = self.send_command(self.ser_tv, cmd_on)
        self.send_command(self.ser_tv, 'exit')
        logging.debug(f"VAC ON 명령 응답: {res}")

        # 5) ON 상태 측정
        self._measure_for_mode(
            mode_label="ON",
            on_finished=self._after_on_measured
        )

    except Exception as e:
        logging.exception(f"[_after_tv_readback] 실패: {e}")


# =========================
# 4) ON 측정 완료 → Diff & 스펙 판정 → 보정 호출
# =========================
def _after_on_measured(self, on_result):
    try:
        self._on_result = on_result

        # 6) Diff 계산 & UI 반영
        spec_ok = self._compute_diff_and_update_tables(self._off_result, self._on_result)

        if spec_ok:
            logging.info("스펙 만족! 보정 불필요.")
            return

        # 7) 자코비안 기반 보정 호출 (여기서 반복 루프 제어)
        self._run_correction_and_remeasure()

    except Exception as e:
        logging.exception(f"[_after_on_measured] 실패: {e}")
        
        
# =========================
# A) OFF/ON 측정 공용 엔트리
# =========================
def _measure_for_mode(self, mode_label: str, on_finished):
    """
    OFF/ON 공용 측정 래퍼.
    기존 자동계측 루프의 일부를 재사용하거나,
    측정 스레드들을 간단히 묶어 결과 딕셔너리로 콜백.
    """
    logging.info(f"[Measure] {mode_label} 측정 시작")

    # 테이블/차트 초기화(필요 시)
    if mode_label == "OFF":
        self._clear_measure_table_for_off()
    else:
        self._clear_measure_table_for_on()

    # 기존 측정 시퀀스 재사용: run_measurement_step 스타일을 재활용
    # 여기선 간단화: 최종 결과를 콜백으로 넘겨주도록 래핑
    self._run_measurement_sequence(
        mode_label=mode_label,
        on_done=on_finished
    )


def _run_measurement_sequence(self, mode_label: str, on_done):
    """
    여러분의 기존 run_measurement_step/MeasureThread 흐름을 감싼 헬퍼.
    - Gamma(W/R/G/B x 0..255), ColorShift 모두 측정
    - main/sub 동시 수집
    - 그래프 갱신: self.ui.vac_chart_gamma_3(상/하), self.ui.vac_chart_colorShift_2
    - 테이블 갱신:
        * OFF: self.ui.vac_table_measure_results_main_2 col 1~3 (Lv,Cx,Cy)
        * ON : self.ui.vac_table_measure_results_main_2 col 4~6 (Lv,Cx,Cy)
    - 완료 시 on_done(result_dict) 호출
    """
    # NOTE: 실제 계측 스레드 클래스/콜백은 프로젝트 내 구현을 그대로 써주세요.
    # 여기서는 "동등한 역할"의 틀만 제공합니다.

    self._measure_acc = {
        "mode": mode_label,
        "gamma_main": [],  # 예: [(pattern, g, Lv, Cx, Cy), ...]
        "gamma_sub":  [],  # 예: [(pattern, g, Lv, Cx, Cy), ...]
        "colorshift_main": [],  # 예: [(patch, x, y), ...]
        "colorshift_sub":  [],
    }

    # ▼▼ 기존 run_measurement_step 처럼 패턴/그레이 루프 돌리며
    # MeasureThread 두 대(main/sub) 결과를 모아서 아래 helper로 넘겨주세요.
    # 여기서는 “측정 결과 처리 콜백”만 보여줍니다.

    def _on_one_gamma_sample(role, pattern, g, xyz_tuple):
        # xyz_tuple -> (x, y, Lv, cct, duv) 가정
        x, y, lv, cct, duv = xyz_tuple
        if role == 'main':
            self._measure_acc["gamma_main"].append((pattern, g, lv, x, y))
            # 그래프(상단) 업데이트
            self._update_gamma_chart(pattern, g, lv, role='main')
            # 테이블 업데이트
            if mode_label == "OFF":
                self._append_main_table_row(lv, x, y, cols=(0,1,2))  # 1~3열
            else:
                self._append_main_table_row(lv, x, y, cols=(3,4,5))  # 4~6열
        else:
            self._measure_acc["gamma_sub"].append((pattern, g, lv, x, y))
            # 그래프(하단) 업데이트
            self._update_gamma_chart(pattern, g, lv, role='sub')

    def _on_one_colorshift_sample(role, patch_name, xy_tuple):
        x, y = xy_tuple
        if role == 'main':
            self._measure_acc["colorshift_main"].append((patch_name, x, y))
        else:
            self._measure_acc["colorshift_sub"].append((patch_name, x, y))
        # 공용 ColorShift 차트 업데이트
        self._update_colorshift_chart(patch_name, x, y, role=role)

    def _on_all_done():
        # 결과 집계 형태를 on_done에 넘김
        result = self._finalize_measurement_result(self._measure_acc)
        on_done(result)

    # 실제 측정 루프 시작 (여러분의 기존 스레드/큐를 사용)
    self._kickoff_measure_threads(
        on_gamma_sample=_on_one_gamma_sample,
        on_colorshift_sample=_on_one_colorshift_sample,
        on_done=_on_all_done
    )


# =========================
# B) TV Read/Write 스레드 래퍼
# =========================
def _write_vac_to_tv(self, vac_data_json: str, on_written):
    """
    기존 WriteVACdataThread를 이용해 JSON을 TV에 write
    """
    self.write_random_VAC_thread = WriteVACdataThread(
        parent=self,
        ser_tv=self.ser_tv,
        vacdataName="TARGET_FROM_DB",
        vacdata_loaded=vac_data_json
    )
    self.write_random_VAC_thread.write_finished.connect(
        lambda ok, msg: on_written(ok, msg)
    )
    self.write_random_VAC_thread.start()


def _read_vac_from_tv(self, on_read):
    """
    기존 ReadVACdataThread를 이용해 TV 현재 LUT read-back
    """
    self.read_random_VAC_thread = ReadVACdataThread(
        parent=self,
        ser_tv=self.ser_tv,
        vacdataName="CURRENT"
    )
    self.read_random_VAC_thread.data_read.connect(
        lambda vac_dict: on_read(vac_dict)
    )
    self.read_random_VAC_thread.error_occurred.connect(
        lambda msg: (logging.error(f"[VAC Read] 실패: {msg}"))
    )
    self.read_random_VAC_thread.start()


# =========================
# C) DB 조회(PanelMaker/FrameRate → VAC_Info_PK → VAC_Data)
# =========================
def _fetch_target_vac_from_db(self):
    panel_name = self.ui.vac_cmb_PanelMaker.currentText()
    frame_rate = self.ui.vac_cmb_FrameRate.currentText()
    conn_params = self.conn_params  # 여러분 앱의 DB 접속정보
    DBName = self.DBName

    db = pymysql.connect(**conn_params)
    try:
        cur = db.cursor()

        # Application status에서 PK 조회
        q1 = f"""
            SELECT VAC_Info_PK 
            FROM {DBName}.W_VAC_Application_Status
            WHERE Panel_Maker=%s AND Frame_Rate=%s
            LIMIT 1
        """
        cur.execute(q1, (panel_name, frame_rate))
        row = cur.fetchone()
        if not row:
            logging.warning("W_VAC_Application_Status에 매칭 행 없음")
            return None, None
        vac_pk = int(row[0])

        # 해당 PK의 VAC_Data 조회
        q2 = f"""
            SELECT VAC_Data
            FROM {DBName}.W_VAC_Info
            WHERE PK=%s
            LIMIT 1
        """
        cur.execute(q2, (vac_pk,))
        row2 = cur.fetchone()
        if not row2:
            logging.warning("W_VAC_Info에서 VAC_Data 조회 실패")
            return None, None

        vac_data_json = row2[0]
        return vac_pk, vac_data_json
    finally:
        db.close()


# =========================
# D) Diff 계산 & 스펙 판정 & 테이블 반영
# =========================
def _compute_diff_and_update_tables(self, off_result, on_result) -> bool:
    """
    스펙:
      |Γ_pred − Γ_ref| ≤ 0.05
      |ΔCx|, |ΔCy|    ≤ 0.003
    여기서는 Γ는 'Lv 기반 Γ'를 이미 구해왔다고 가정하거나,
    필요시 오프라인 참조 Γ(ref)와 ON의 Γ를 계산해서 비교하도록 연결하세요.
    우선 Cx/Cy Diff를 명시적으로 테이블 9~10열에 기록.
    """
    # 예시: 동일 순서로 저장됐다고 가정하고 pairwise diff
    # (실제 앱에서는 (pattern, gray) 키로 join해서 안전하게 매칭하세요)
    off_main = off_result.get("gamma_main", [])
    on_main  = on_result.get("gamma_main", [])

    # 안전장치
    n = min(len(off_main), len(on_main))
    if n == 0:
        logging.warning("Diff 계산 대상이 없습니다.")
        return False

    # 테이블 갱신을 위해 현재 row 인덱스를 기억
    # (실제 구현에선 insert시 저장해둔 row index를 같이 들고 다니는 게 가장 안전)
    for i in range(n):
        p_off, g_off, lv_off, cx_off, cy_off = off_main[i]
        p_on,  g_on,  lv_on,  cx_on,  cy_on  = on_main[i]

        # 안전 매칭 검사
        if (p_off != p_on) or (g_off != g_on):
            logging.debug(f"키 불일치: OFF=({p_off},{g_off}) vs ON=({p_on},{g_on})")
            continue

        dCx = float(cx_on - cx_off)
        dCy = float(cy_on - cy_off)

        # 테이블: 9~10열에 diff
        row = i  # 간단화(여러분 앱에서는 행 인덱스 매핑을 정확히 사용)
        _set_tbl = self.ui.vac_table_measure_results_main_2.setItem
        _set_tbl(row, 8,  QTableWidgetItem(f"{dCx:.6f}"))  # 9열 (0-index=8)
        _set_tbl(row, 9,  QTableWidgetItem(f"{dCy:.6f}"))  # 10열 (0-index=9)

    # 간단 스펙 체크(최대 절대값)
    dCx_all = [float(on_main[i][3] - off_main[i][3]) for i in range(n)]
    dCy_all = [float(on_main[i][4] - off_main[i][4]) for i in range(n)]

    cx_ok = (np.max(np.abs(dCx_all)) <= 0.003) if dCx_all else False
    cy_ok = (np.max(np.abs(dCy_all)) <= 0.003) if dCy_all else False

    # Γ 스펙(0.05)은 여러분의 Γ 계산 루틴/참조값에 연결해주세요.
    gamma_ok = True  # 자리표시자

    spec_ok = (cx_ok and cy_ok and gamma_ok)
    logging.info(f"[SPEC] Cx:{cx_ok} Cy:{cy_ok} Gamma:{gamma_ok} → Overall:{spec_ok}")
    return spec_ok


# =========================
# E) 자코비안 보정 훅
# =========================
def _run_correction_and_remeasure(self):
    """
    OFF vs ON Diff → 자코비안 보정 → LUT 업데이트 → Read-back → ON 재측정 → 반복/종료
    여기서는 보정 함수만 호출하고, 그 결과 LUT를 TV에 적용하는 루프를 형식만 둡니다.
    """
    try:
        # 1) 자코비안 보정 (여기에 여러분이 만든 pkl로드/보정 함수를 연결하세요)
        #    예: new_high = run_jacobian_correction(off=self._off_result, on=self._on_result, ...)
        #    반환: lut_dict = {"R_Low":..., "R_High":..., ...} (Low는 원본 유지, High만 변경)
        lut_dict = self.run_jacobian_correction(self._off_result, self._on_result)

        # 2) TV 적용 → Read-back 확인
        self._write_vac_to_tv(
            vac_data_json=json.dumps(lut_dict),
            on_written=lambda ok, msg: (
                self._read_vac_from_tv(on_read=lambda _: self._measure_for_mode("ON", self._after_on_measured))
                if ok else logging.warning(f"[Correction Write] 실패: {msg}")
            )
        )
        # 이후 self._after_on_measured에서 스펙 만족/반복 탈출 관리

    except Exception as e:
        logging.exception(f"[_run_correction_and_remeasure] 실패: {e}")
        
def _update_gamma_chart(self, pattern, gray, lv, role='main'):
    """ 여러분의 self.update_measure_results_to_graph 호출로 대체해도 됩니다. """
    try:
        self.update_measure_results_to_graph(gray, lv, role=role, pattern=pattern)
    except Exception as e:
        logging.debug(f"_update_gamma_chart warn: {e}")

def _update_colorshift_chart(self, patch_name, x, y, role='main'):
    """ ColorShift 공용 차트 갱신: self.ui.vac_chart_colorShift_2 """
    try:
        # 여러분의 colorshift 차트 업데이트 함수 호출
        # 예: self.update_colorshift_chart(self.ui.vac_chart_colorShift_2, patch_name, x, y, role)
        pass
    except Exception as e:
        logging.debug(f"_update_colorshift_chart warn: {e}")

def _append_main_table_row(self, lv, cx, cy, cols=(0,1,2)):
    """
    self.ui.vac_table_measure_results_main_2 에 한 줄 추가/갱신
    cols: (Lv, Cx, Cy)를 기록할 0-index 열 번호 튜플
    """
    tbl = self.ui.vac_table_measure_results_main_2
    row = tbl.rowCount()
    tbl.insertRow(row)
    tbl.setItem(row, cols[0], QTableWidgetItem(f"{float(lv):.6f}"))
    tbl.setItem(row, cols[1], QTableWidgetItem(f"{float(cx):.6f}"))
    tbl.setItem(row, cols[2], QTableWidgetItem(f"{float(cy):.6f}"))

def _clear_measure_table_for_off(self):
    self.ui.vac_table_measure_results_main_2.setRowCount(0)

def _clear_measure_table_for_on(self):
    # ON 결과는 같은 표의 4~6열, Diff는 9~10열에 기록하므로
    # 보존하려면 행 삭제 없이 추가 열만 채우는 방식도 가능. 우선은 유지.
    pass

def _finalize_measurement_result(self, acc_dict):
    """
    측정 누적(acc_dict)을 최종 결과 포맷으로 정리
    필요시 (pattern, gray) 키로 정렬/정규화
    """
    # 여기서는 그대로 반환
    return acc_dict