# -*- coding: utf-8 -*-
import time, json, logging
import numpy as np
import pandas as pd
import joblib

from PySide2.QtCore import QObject, Signal, Slot, QEventLoop, QTimer
# 위젯은 기존 UI 멤버(self.ui....) 그대로 사용

# =============== 유틸: 수치/보간/자코비안 ===============

def compute_gamma_from_lv(nor_lv: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    nor_lv[g] (0~1 정규화된 Lv)로부터 감마 계산 (g=1..254 유효)
    Γ(g) = log(nor_lv[g]) / log(g/255)
    g=0,255 또는 nor_lv<=0 은 NaN
    """
    L = len(nor_lv)
    g = np.arange(L, dtype=np.float32)
    gray_norm = np.clip(g / 255.0, eps, 1.0)  # 0→eps
    gamma = np.full(L, np.nan, np.float32)

    mask = (g > 0) & (g < 255) & (nor_lv > eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma[mask] = np.log(nor_lv[mask]) / np.log(gray_norm[mask])
    return gamma

def linear_interp_weights_1d(x, x0, x1):
    # 0..1 선형 보간 가중치
    if x1 == x0: 
        return 1.0, 0.0
    t = (x - x0) / float(x1 - x0)
    return (1.0 - t), t

def resample_curve_linear(src: np.ndarray, dst_len: int) -> np.ndarray:
    """
    src: 길이 N (예: 256) → dst_len (예: 4096) 선형보간
    """
    N = len(src)
    if dst_len == N:
        return src.copy()
    out = np.empty(dst_len, np.float32)
    for i in range(dst_len):
        x = (i / (dst_len - 1.0)) * (N - 1.0)
        i0 = int(np.floor(x))
        i1 = min(N - 1, i0 + 1)
        w0, w1 = linear_interp_weights_1d(x, i0, i1)
        out[i] = src[i0] * w0 + src[i1] * w1
    return out

def enforce_monotone_inplace(arr: np.ndarray):
    np.maximum.accumulate(arr, out=arr)

def clamp01_inplace(arr: np.ndarray):
    np.clip(arr, 0.0, 1.0, out=arr)

def stack_basis_all_grays(knots_idx: np.ndarray, L=256) -> np.ndarray:
    """
    knot 인덱스(0..255) 기반 모자(hat) 기저행렬 Φ (LxK)
    """
    K = len(knots_idx)
    Phi = np.zeros((L, K), np.float32)
    for g in range(L):
        # 경계
        if g <= knots_idx[0]:
            Phi[g, 0] = 1.0
            continue
        if g >= knots_idx[-1]:
            Phi[g, -1] = 1.0
            continue
        i = np.searchsorted(knots_idx, g) - 1
        g0, g1 = knots_idx[i], knots_idx[i+1]
        denom = max(1, (g1 - g0))
        t = (g - g0) / denom
        Phi[g, i]   = 1.0 - t
        Phi[g, i+1] = t
    return Phi

def build_A_from_artifacts(artifacts: dict, comp: str) -> np.ndarray:
    """
    ΔY ≈ A Δh, 여기서 Δh = [Δh_R(K), Δh_G(K), Δh_B(K)] (총 3K)
    A = [Φ * diag(β_R) | Φ * diag(β_G) | Φ * diag(β_B)]
    """
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    Phi = stack_basis_all_grays(knots, L=256)  # (256, K)

    comp_obj = artifacts["components"][comp]
    coef = np.asarray(comp_obj["coef"], dtype=np.float32)
    s = comp_obj["feature_slices"]
    sR = slice(s["high_R"][0], s["high_R"][1])
    sG = slice(s["high_G"][0], s["high_G"][1])
    sB = slice(s["high_B"][0], s["high_B"][1])

    beta_R = coef[sR]  # (K,)
    beta_G = coef[sG]
    beta_B = coef[sB]

    A_R = Phi * beta_R.reshape(1, -1)
    A_G = Phi * beta_G.reshape(1, -1)
    A_B = Phi * beta_B.reshape(1, -1)
    A = np.hstack([A_R, A_G, A_B]).astype(np.float32)  # (256, 3K)
    return A

def solve_delta_h(A: np.ndarray, dY: np.ndarray, ridge_lambda=1e-3, max_step=0.05):
    """
    Ridge 해: Δh = (A^T A + λI)^-1 A^T dY
    채널별(K,R/G/B) 묶음 반환: Δh_R, Δh_G, Δh_B (각 K,)
    max_step: knot 변화량 제한(안정화)
    """
    # 정방형 해
    AT = A.T
    ATA = AT @ A
    b = AT @ dY
    # (ATA + λI) x = b
    K3 = ATA.shape[0]
    ATA.flat[::K3+1] += ridge_lambda
    delta_h = np.linalg.solve(ATA, b).astype(np.float32)

    # 안정화: 과도 변화 clamp
    delta_h = np.clip(delta_h, -max_step, max_step)
    return delta_h

def delta_h_to_delta_high256(delta_h: np.ndarray, artifacts: dict) -> dict:
    """
    Δh(3K,) → 각 채널 ΔHigh(256,) 생성
    """
    knots = np.asarray(artifacts["knots"], dtype=np.int32)
    Phi = stack_basis_all_grays(knots, L=256)  # (256, K)
    K = len(knots)

    dR = Phi @ delta_h[:K]
    dG = Phi @ delta_h[K:2*K]
    dB = Phi @ delta_h[2*K:3*K]

    return {"R_High": dR.astype(np.float32),
            "G_High": dG.astype(np.float32),
            "B_High": dB.astype(np.float32)}

# =============== 유틸: QThread 동기화 래퍼 ===============

def wait_thread_result(start_thread_fn, on_success_signal, on_error_signal=None, timeout_ms=600000):
    """
    QThread 기반 비동기 → 동기처럼 기다리기
    start_thread_fn(): thread.start()를 내부에서 호출
    on_success_signal: Signal(object or tuple)
    on_error_signal: (optional) Signal(str)
    timeout_ms: 타임아웃
    return: (ok, payload_or_msg)
    """
    loop = QEventLoop()
    payload_box = {"ok": False, "data": None}

    def _on_ok(*args):
        payload_box["ok"] = True
        payload_box["data"] = args[0] if len(args) == 1 else args
        loop.quit()

    def _on_err(msg):
        payload_box["ok"] = False
        payload_box["data"] = msg
        loop.quit()

    on_success_signal.connect(_on_ok)
    if on_error_signal is not None:
        on_error_signal.connect(_on_err)

    start_thread_fn()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_()

    # disconnect는 생략(짧은 루틴)
    return payload_box["ok"], payload_box["data"]

# =============== 메인 플로우 메서드들 ===============

def vac_luna_send(self, cmd: str) -> str:
    self.send_command(self.ser_tv, 's')
    res = self.send_command(self.ser_tv, cmd)
    self.send_command(self.ser_tv, 'exit')
    return res

def vac_off_if_needed(self) -> bool:
    st = self.check_VAC_status()  # {'supported':..., 'activated':..., 'vacdata':...}
    if not st.get("supported", False):
        logging.info("VAC 미지원 모델. OFF 측정만 진행합니다.")
        return True
    if st.get("activated", False):
        logging.debug("VAC 활성 상태 감지 → VAC OFF 시도")
        cmd = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/setVACActive \'{"OnOff":false}\''
        res = vac_luna_send(self, cmd)
        logging.debug(f"[VAC OFF] 응답: {res}")
        st2 = self.check_VAC_status()
        if st2.get("activated", False):
            logging.warning("VAC OFF 실패로 보입니다. 그래도 측정은 진행합니다.")
        else:
            logging.info("VAC OFF 전환 성공.")
    else:
        logging.debug("이미 VAC OFF 상태.")
    return True

def measure_gamma_colorshift_once(self, label_prefix:str):
    """
    한 번 측정(감마 0~255, 4패턴; 컬러쉬프트 12패턴) 수행.
    - 기존 run_measurement_step()의 스레드 흐름을 동기 래퍼로 호출했다고 가정
    - 결과: dict 형태로 표준화 반환 (패턴별 Lv(256), Cx(256), Cy(256), 그리고 Macbeth Cx/Cy)
    UI 업데이트(차트/테이블)는 질문에서 주신 기존 메서드들을 호출해줍니다.
    """
    # 여기서는 여러분이 이미 가진 스레드/업데이트 함수를 그대로 사용한다고 가정하고,
    # 간단한 blocking 래퍼만 제공합니다.
    results_box = {"gamma": {}, "colorshift": {}}  # gamma: {pattern:{'Lv', 'Cx','Cy'}}, colorshift:{patch:{'Cx0','Cy0','Cx60','Cy60'}}
    # ---- 여러분의 기존 트리거 사용 ----
    # main & sub 모두 돌도록 기존 run_measurement_step를 한 라운드 트리거하는 helper를 쓰세요.
    # 아래 두 줄은 예시용이며, 실제론 여러분의 메서드를 동기 래핑해서 결과를 모아야 합니다.
    # 이 부분은 프로젝트의 스레드/시그널 구조에 강하게 의존하므로, 이미 쓰시던 것을 그대로 호출하세요.
    # 결과만 아래 형태로 채우면, 후속 단계(자코비안/검증)는 그대로 동작합니다.
    #
    # results_box["gamma"]["W"] = {"Lv": np.array([...], np.float32), "Cx":..., "Cy":...}
    # results_box["gamma"]["R"] = {...}
    # ...
    # results_box["colorshift"]["Darkskin"] = {"Cx0":..., "Cy0":..., "Cx60":..., "Cy60":...}
    #
    # --- 예시(빈 틀) ---
    raise NotImplementedError("여기서 기존 측정 스레드 호출해 results_box를 채우세요.")
    # return results_box

def build_deltaY_from_two_measurements(off_meas: dict, on_meas: dict, patterns_used=('W',)):
    """
    ΔY 타깃 생성: (ref=OFF) - (pred=ON)
    components: Gamma, Cx, Cy (패턴은 W만 사용 권장)
    반환 dict: {'Gamma': (256,), 'Cx':(256,), 'Cy':(256,)}
    """
    d = {}
    for comp in ("Gamma","Cx","Cy"):
        # W만 사용(다른 패턴 확장 가능)
        p = patterns_used[0]
        if comp == "Gamma":
            # Gamma는 Lv로부터 계산
            gamma_off = compute_gamma_from_lv(off_meas["gamma"][p]["Lv"])
            gamma_on  = compute_gamma_from_lv(on_meas["gamma"][p]["Lv"])
            diff = np.nan_to_num(gamma_off - gamma_on, nan=0.0)
        else:
            v_off = np.asarray(off_meas["gamma"][p][comp], dtype=np.float32)
            v_on  = np.asarray(on_meas["gamma"][p][comp],  dtype=np.float32)
            diff = np.nan_to_num(v_off - v_on, nan=0.0)
        d[comp] = diff.astype(np.float32)
    return d

def apply_delta_high_256_to_4096_and_write(self, cur_lut_4096: dict, delta_high_256: dict):
    """
    ΔHigh(256)을 4096으로 보간해서 LUT High 갱신 → TV에 적용 + 검증 읽기 + UI 업데이트
    cur_lut_4096: {'R_Low','R_High','G_Low','G_High','B_Low','B_High'} (각 4096,)
    """
    # 1) 256→4096 보간
    for ch in ("R_High","G_High","B_High"):
        d256 = delta_high_256[ch]
        d4096 = resample_curve_linear(d256, 4096)
        cur_lut_4096[ch] = (cur_lut_4096[ch] + d4096).astype(np.float32)
        clamp01_inplace(cur_lut_4096[ch])
        enforce_monotone_inplace(cur_lut_4096[ch])

    # 2) TV에 write (여러분의 WriteVACdataThread 사용)
    def _start_write():
        self.write_random_VAC_thread = WriteVACdataThread(
            parent=self,
            ser_tv=self.ser_tv,
            vacdataName=self.vacdataName,
            vacdata_loaded={
                "R_Low":  cur_lut_4096["R_Low"].tolist(),
                "R_High": cur_lut_4096["R_High"].tolist(),
                "G_Low":  cur_lut_4096["G_Low"].tolist(),
                "G_High": cur_lut_4096["G_High"].tolist(),
                "B_Low":  cur_lut_4096["B_Low"].tolist(),
                "B_High": cur_lut_4096["B_High"].tolist(),
            }
        )
        self.write_random_VAC_thread.write_finished.connect(lambda ok, msg: None)  # 시그널 연결은 wait에서 함
        self.write_random_VAC_thread.start()

    ok, data = wait_thread_result(
        start_thread_fn=_start_write,
        on_success_signal=self.write_random_VAC_thread.write_finished
    )
    if not ok:
        logging.error(f"[LUT Write] 실패: {data}")
        return False

    # 3) Read로 검증 + UI 업데이트 (self.ui.vac_graph_rgbLUT_4, self.ui.vac_table_rbgLUT_4)
    def _start_read():
        self.read_random_VAC_thread = ReadVACdataThread(parent=self, ser_tv=self.ser_tv, vacdataName=self.vacdataName)
        self.read_random_VAC_thread.data_read.connect(lambda payload: None)
        self.read_random_VAC_thread.error_occurred.connect(lambda msg: None)
        self.read_random_VAC_thread.start()

    ok2, lut_read = wait_thread_result(
        start_thread_fn=_start_read,
        on_success_signal=self.read_random_VAC_thread.data_read,
        on_error_signal=self.read_random_VAC_thread.error_occurred
    )
    if not ok2:
        logging.warning(f"[LUT Read] 실패 또는 미일치: {lut_read}")
    else:
        # 차트/테이블 업데이트 (여러분의 기존 함수)
        try:
            channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']
            rgb_df = pd.DataFrame({ch: lut_read.get(ch.replace("_","channel"), []) for ch in channels})
            self.update_rgbchannel_chart(
                rgb_df,
                self.graph['vac_laboratory']['data_acquisition_system']['input']['ax'],
                self.graph['vac_laboratory']['data_acquisition_system']['input']['canvas']
            )
            self.update_rgbchannel_table(rgb_df, self.ui.vac_table_rbgLUT_4)
        except Exception as e:
            logging.warning(f"[LUT UI 업데이트] {e}")
    return True

def meets_spec(off_meas: dict, on_meas: dict, thr_gamma=0.05, thr_c=0.003) -> bool:
    """
    스펙 판정:
      max |Gamma_on - Gamma_off| ≤ thr_gamma
      max |Cx_on - Cx_off| ≤ thr_c
      max |Cy_on - Cy_off| ≤ thr_c
    (W 패턴 기준)
    """
    p = 'W'
    g_off = compute_gamma_from_lv(off_meas["gamma"][p]["Lv"])
    g_on  = compute_gamma_from_lv(on_meas["gamma"][p]["Lv"])
    dG = np.nan_to_num(np.abs(g_on - g_off), nan=0.0).max()

    dCx = np.abs(np.asarray(on_meas["gamma"][p]["Cx"]) - np.asarray(off_meas["gamma"][p]["Cx"])).max()
    dCy = np.abs(np.asarray(on_meas["gamma"][p]["Cy"]) - np.asarray(off_meas["gamma"][p]["Cy"])).max()

    logging.info(f"[Spec] max|ΔGamma|={dG:.4f}, max|ΔCx|={dCx:.4f}, max|ΔCy|={dCy:.4f}")
    return (dG <= thr_gamma) and (dCx <= thr_c) and (dCy <= thr_c)

# =============== 메인 엔트리: 버튼 이벤트 연결용 ===============

def start_VAC_optimization(self):
    """
    전체 플로우:
      1) VAC OFF 보장 → 측정(OFF baseline) + UI 업데이트
      2) DB에서 모델/주사율 매칭 VAC Data 가져와 TV에 적용(ON) → 측정(ON 현재) + UI 업데이트
      3) 스펙 확인 → 통과면 종료
      4) 미통과면 자코비안 기반 보정(256기준) → 4096 보간 반영 → TV 적용 → 재측정 → 스펙 재확인
      5) (필요 시 반복 2~3회만)
    """
    try:
        # (0) 자코비안 로드
        jac_path = os.path.join(self.scripts_dir, "jacobian_Y0_high.pkl")
        artifacts = joblib.load(jac_path)

        A_Gamma = build_A_from_artifacts(artifacts, "Gamma")  # (256, 3K)
        A_Cx    = build_A_from_artifacts(artifacts, "Cx")
        A_Cy    = build_A_from_artifacts(artifacts, "Cy")

        # (1) VAC OFF 보장 + 측정
        self.vac_btn_startOptimization.setEnabled(False)
        vac_off_if_needed(self)

        logging.info("[STEP1] VAC OFF 상태 측정 시작")
        off_meas = measure_gamma_colorshift_once(self, label_prefix="[OFF]")  # <-- 여러분 측정 래퍼 구현 필요

        # (2) DB에서 모델/주사율 매칭 VAC Data → TV 적용 → 읽기 → UI 업데이트
        logging.info("[STEP2] DB에서 적용 VAC 데이터 가져와 TV 적용")
        panel_txt = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fps_txt   = self.ui.vac_cmb_FrameRate.currentText().strip()

        # DB 조회 (사용자 함수로 구현돼 있다고 가정)
        vac_pk, vac_payload_4096 = self.lookup_vac_data_from_db(panel_txt, fps_txt)  # (pk, dict of arrays len=4096)
        if vac_payload_4096 is None:
            logging.warning("매칭되는 VAC 데이터가 없어 기본 LUT 유지 후 진행합니다.")
            cur_lut_4096 = self.read_current_tv_lut_4096()  # 사용자 제공
        else:
            # TV에 write + read + UI
            cur_lut_4096 = vac_payload_4096.copy()
            ok_write = apply_delta_high_256_to_4096_and_write(self, cur_lut_4096, 
                          {"R_High": np.zeros(256, np.float32),
                           "G_High": np.zeros(256, np.float32),
                           "B_High": np.zeros(256, np.float32)})
            if not ok_write:
                logging.warning("초기 VAC 적용 실패. 현재 TV LUT를 읽어서 사용합니다.")
                cur_lut_4096 = self.read_current_tv_lut_4096()

        # (3) VAC ON 측정
        logging.info("[STEP3] VAC ON 상태 측정 시작")
        on_meas = measure_gamma_colorshift_once(self, label_prefix="[ON]")   # <-- 여러분 측정 래퍼 구현 필요

        # (4) 스펙 판정
        if meets_spec(off_meas, on_meas, thr_gamma=0.05, thr_c=0.003):
            logging.info("✅ 스펙 만족. 최적화 종료.")
            return

        # (5) 자코비안 보정 루프(최대 2~3회 권장)
        for it in range(3):
            logging.info(f"[STEP4] 자코비안 보정 {it+1}/3")

            # ΔY 구성 (OFF - ON) — W 패턴 기준
            deltaY = build_deltaY_from_two_measurements(off_meas, on_meas, patterns_used=('W',))

            # 연립: comp별로 Δh 구해 평균/가중 결합 (여기선 동일 가중 평균)
            dH_list = []
            for A, comp in ((A_Gamma, "Gamma"), (A_Cx, "Cx"), (A_Cy, "Cy")):
                dY = deltaY[comp]  # (256,)
                delta_h = solve_delta_h(A, dY, ridge_lambda=2e-3, max_step=0.05)  # 안전한 범위
                dH_256 = delta_h_to_delta_high256(delta_h, artifacts)  # dict of 3x(256,)
                dH_list.append(dH_256)

            # 평균결합
            dH_mean = {
                "R_High": np.mean([d["R_High"] for d in dH_list], axis=0).astype(np.float32),
                "G_High": np.mean([d["G_High"] for d in dH_list], axis=0).astype(np.float32),
                "B_High": np.mean([d["B_High"] for d in dH_list], axis=0).astype(np.float32),
            }

            # 적용(4096 보간 → LUT High 업데이트 → TV write/read/UI)
            ok = apply_delta_high_256_to_4096_and_write(self, cur_lut_4096, dH_mean)
            if not ok:
                logging.warning("보정 LUT 적용 실패. 루프 중단.")
                break

            # 재측정
            on_meas = measure_gamma_colorshift_once(self, label_prefix=f"[ON after JAC {it+1}]")

            # 스펙 재판정
            if meets_spec(off_meas, on_meas, thr_gamma=0.05, thr_c=0.003):
                logging.info("✅ 보정 후 스펙 만족. 종료합니다.")
                break
        else:
            logging.info("ℹ️ 최대 보정 회수 도달. 현재 상태로 종료합니다.")

    except NotImplementedError as e:
        logging.error(f"[구현 필요] {e}")
    except Exception as e:
        logging.exception(e)
    finally:
        self.vac_btn_startOptimization.setEnabled(True)