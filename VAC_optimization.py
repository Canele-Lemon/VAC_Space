    def start_VAC_optimization(self):
        
        
        # 1. VAC Control Off 하기 : VAC Control status 확인 후 On 이면 Off 후 측정 / Off 이면 바로 측정
        st = self.check_VAC_status()
        if st.get("activated", False):
            logging.debug("VAC 활성 상태 감지 → VAC OFF 시도")
            cmd = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff":false}\''
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
        
        # 2. VAC OFF일 때 측정
        
        # 3. VAC ON하고 DB에서 모델정보+패널주사율에 해당하는 VAC Data 가져와서 TV 적용 후 측정
        
        # 4. 보정 로직(자코비안 기반) 호출
        
        # 5. 재측정 (검증)
        
        
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
