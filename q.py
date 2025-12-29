import sys
from pathlib import Path
import numpy as np
from PySide2.QtCore import QThread, Signal

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subpages.vacspace_130.src.policy.vac_spec_policy import VACSpecPolicy
class SpecEvalThread(QThread):
    finished = Signal(bool, dict)  # (spec_ok, metrics)

    def __init__(self, off_store, on_store, policy: VACSpecPolicy, parent=None):
        super().__init__(parent)
        if policy is None:
            raise ValueError("policy must be provided (VACSpecPolicy)")
        self.off_store = off_store
        self.on_store  = on_store
        self.policy    = policy

    @staticmethod
    def _compute_gamma_series(lv_vec_256):
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

    @staticmethod
    def _extract_white(series_store):
        lv = np.full(256, np.nan, np.float64)
        cx = np.full(256, np.nan, np.float64)
        cy = np.full(256, np.nan, np.float64)
        for g in range(256):
            tup = series_store['gamma']['main']['white'].get(g, None)
            if tup:
                lv[g], cx[g], cy[g] = float(tup[0]), float(tup[1]), float(tup[2])
        return lv, cx, cy

    @staticmethod
    def compute_gray_errors_and_ng_list(off_store, on_store, policy: VACSpecPolicy):
        # 1) OFF/ON 시리즈
        lv_off, cx_off, cy_off = SpecEvalThread._extract_white(off_store)
        lv_on , cx_on , cy_on  = SpecEvalThread._extract_white(on_store)

        # 2) Gamma series
        G_off = SpecEvalThread._compute_gamma_series(lv_off)
        G_on  = SpecEvalThread._compute_gamma_series(lv_on)

        # 3) Δ = ON - OFF
        dG  = (G_on  - G_off).astype(np.float64)
        dCx = (cx_on - cx_off).astype(np.float64)
        dCy = (cy_on - cy_off).astype(np.float64)

        abs_dG  = np.abs(dG)
        abs_dCx = np.abs(dCx)
        abs_dCy = np.abs(dCy)

        # 4) policy 기반 eval grays / NG 산출 (여기서만!)
        gamma_eval_grays = sorted(policy.gamma_eval_grays)
        color_eval_grays = sorted(policy.color_eval_grays)

        ng_grays_gamma = []
        for g in gamma_eval_grays:
            if np.isfinite(abs_dG[g]) and (abs_dG[g] > policy.thr_gamma):
                ng_grays_gamma.append(g)

        ng_grays_color = []
        for g in color_eval_grays:
            bad_cx = (np.isfinite(abs_dCx[g]) and abs_dCx[g] > policy.thr_c)
            bad_cy = (np.isfinite(abs_dCy[g]) and abs_dCy[g] > policy.thr_c)
            if bad_cx or bad_cy:
                ng_grays_color.append(g)

        ng_grays = sorted(set(ng_grays_gamma) | set(ng_grays_color))

        return dG, dCx, dCy, ng_grays, ng_grays_gamma, ng_grays_color, gamma_eval_grays, color_eval_grays

    def run(self):
        try:
            pol = self.policy
            dG, dCx, dCy, ng_grays, ng_gamma, ng_color, gamma_eval_grays, color_eval_grays = \
                self.compute_gray_errors_and_ng_list(self.off_store, self.on_store, pol)
                            
            abs_dG  = np.abs(dG)
            abs_dCx = np.abs(dCx)
            abs_dCy = np.abs(dCy)
            
            if gamma_eval_grays:
                max_dG = float(np.nanmax(abs_dG[gamma_eval_grays]))
            else:
                max_dG = float("nan")

            if color_eval_grays:
                max_dCx = float(np.nanmax(abs_dCx[color_eval_grays]))
                max_dCy = float(np.nanmax(abs_dCy[color_eval_grays]))
            else:
                max_dCx = float("nan")
                max_dCy = float("nan")

            spec_ok = (len(ng_grays) == 0)

            metrics = {
                "max_dG":  max_dG,
                "max_dCx": max_dCx,
                "max_dCy": max_dCy,

                "thr_gamma": pol.thr_gamma,
                "thr_c": pol.thr_c,

                # raw vectors
                "dG":  dG.astype(np.float32),
                "dCx": dCx.astype(np.float32),
                "dCy": dCy.astype(np.float32),

                # NG / eval grays
                "ng_grays": ng_grays,
                "ng_grays_gamma": ng_gamma,
                "ng_grays_color": ng_color,
                "gamma_eval_grays": gamma_eval_grays,
                "color_eval_grays": color_eval_grays,
            }
            self.finished.emit(spec_ok, metrics)

        except Exception:
            self.finished.emit(False, {"error": True})

@dataclass(frozen=True)
class VACSpecPolicy:
    thr_gamma: float = 0.05
    thr_c: float = 0.003

    gamma_eval_grays: FrozenSet[int] = frozenset(range(2, 248))
    color_eval_grays: FrozenSet[int] = frozenset(range(6, 256))
    
    def should_eval_gamma(self, gray: int) -> bool:
        return gray in self.gamma_eval_grays

    def should_eval_color(self, gray: int) -> bool:
        return gray in self.color_eval_grays

    def gamma_ok(self, d_g: float) -> bool:
        return abs(d_g) <= self.thr_gamma

    def color_ok(self, d_cx: float, d_cy: float) -> bool:
        return (abs(d_cx) <= self.thr_c) and (abs(d_cy) <= self.thr_c)

이런식으로 spec 정책은 모두 VACSpecPolicy에서 관리하는 것으로 수정했잖아요, 이에 따라 현재 아래까지 수정된 최적화 루프를 어떻게 수정하면 되는지 알려주세요:
    def start_VAC_optimization(self):
        self._spec_policy = VACSpecPolicy()
        
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        
        self._fine_mode = False
        self._fine_ng_list = None
        
        self._load_jacobian_bundle_npy()
        self._load_prediction_models()
        
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")
        
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self._measure_off_ref_then_on()

    def _measure_off_ref_then_on(self):
        profile_off = SessionProfile(
            session_mode="VAC OFF",
            cie_label="data_1",
            table_cols={"lv":0, "cx":1, "cy":2, "gamma":3},
            ref_store=None
        )

        def _after_off(store_off):
            self._off_store = store_off
            lv_off = np.zeros(256, dtype=np.float64)
            for g in range(256):
                tup = store_off['gamma']['main']['white'].get(g, None)
                lv_off[g] = float(tup[0]) if tup else np.nan
            self._gamma_off_vec = self._compute_gamma_series(lv_off)
            
            self._lv_off_vec = lv_off.copy()
            try:
                self._lv_off_max = float(np.nanmax(lv_off[1:]))
            except (ValueError, TypeError):
                self._lv_off_max = float('nan')
            
            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
            logging.info("[TV Control] VAC ON 전환 성공")
            
            logging.info("[Measurement] VAC ON 측정 시작")
            self._apply_predicted_vac_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off,
            on_done=_after_off
        )

    def _apply_predicted_vac_and_measure_on(self):
        self._step_start(2)
        
        BASE_VAC_PK = 3025
        vac_version, base_vac_data = self._fetch_vac_by_vac_info_pk(BASE_VAC_PK)
        if base_vac_data is None:
            logging.error("[DB] VAC 데이터 로딩 실패 - 최적화 루프 종료")
            return

        base_vac_dict = json.loads(base_vac_data)
        self._vac_dict_cache = base_vac_dict
        
        try:
            predicted_vac_data, debug_info = self._generate_predicted_vac_lut(
                base_vac_dict,
                n_iters=2,
                wG=0.4,
                wC=1.0,
                lambda_ridge=1e-3
            )
        except Exception:
            logging.exception("[PredictOpt] 예측 기반 1st 보정 중 예외 발생 - Base VAC로 진행")
            predicted_vac_data, debug_info = None, None
            predicted_vac_data = base_vac_data
            
        predicted_vac_dict = json.loads(predicted_vac_data)
        self._vac_dict_cache = predicted_vac_dict
            
        lut_dict_plot = {key.replace("channel", "_"): v for key, v in predicted_vac_dict.items() if "channel" in key}
        self._update_lut_chart_and_table(lut_dict_plot)
        self._step_done(2)

        def _after_write(ok, msg):
            if not ok:
                logging.error(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 실패: {msg} - 최적화 루프 종료")
                return
            
            logging.info(f"[VAC Writing] 예측 기반 최적화 VAC 데이터 Writing 완료: {msg}")
            logging.info("[VAC Reading] VAC Reading 시작")
            self._read_vac_from_tv(_after_read)

        def _after_read(read_vac_dict):
            self.send_command(self.ser_tv, 'exit')
            if not read_vac_dict:
                logging.error("[VAC Reading] VAC Reading 실패 - 최적화 루프 종료")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=predicted_vac_dict, read_data=read_vac_dict)

            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")

            self._step_done(3)

            self._fine_mode = False
            
            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_on = SessionProfile(
                session_mode="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_on(store_on):
                logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 완료")
                self._step_done(4)
                self._on_store = store_on
                self._update_last_on_lv_norm(store_on)
                
                logging.info("[Evaluation] ΔCx / ΔCy / ΔGamma의 Spec 만족 여부를 평가합니다.")
                self._step_start(5)
                self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
                self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx=0, max_iters=1))
                self._spec_thread.start()

            logging.info("[Measurement] 예측 기반 최적화 VAC 데이터 기준 측정 시작")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_on,
                on_done=_after_on
            )

        logging.info("[VAC Writing] 예측기반 최적화 VAC 데이터 TV Writing 시작")
        self._write_vac_to_tv(predicted_vac_data, on_finished=_after_write)

    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        """
        조건 1) spec_ok==True: 종료
        조건 2) (spec_ok==False) and (iter_idx <= max_iters): NG Gray Correction (batch) 반복
        """
        try:
            pol = getattr(self, "_spec_policy", None)
            if pol is None:
                pol = VACSpecPolicy()
                self._spec_policy = pol
            
            # logging    
            ng_grays = []
            thr_g = None
            thr_c = None
            
            if metrics and "error" not in metrics:
                max_dG   = metrics.get("max_dG",  float("nan"))
                max_dCx  = metrics.get("max_dCx", float("nan"))
                max_dCy  = metrics.get("max_dCy", float("nan"))
                thr_g    = metrics.get("thr_gamma", self._spec_thread.thr_gamma if self._spec_thread else None)
                thr_c    = metrics.get("thr_c",     self._spec_thread.thr_c     if self._spec_thread else None)
                ng_grays = metrics.get("ng_grays", [])
                
                logging.info(
                    f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{thr_g}), "
                    f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c}), "
                    f"NG grays={ng_grays}"
                )
            else:
                logging.warning("[Evaluation] evaluation failed — treating as not passed.")
                ng_grays = []

            self._update_spec_views(iter_idx, self._off_store, self._on_store)

            # 조건 1) spec_ok==True: 종료
            if spec_ok:
                self._step_done(5)
                logging.info("[Evaluation] Spec 통과 — 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            # 조건 2) (spec_ok==False) and (max_iters>0): NG Gray Correction
            self._step_fail(5)
            
            if max_iters <= 0:
                logging.info("[Evaluation] Spec NG but no further correction (max_iters≤0) - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            if iter_idx >= max_iters:
                logging.info("[Evaluation] Spec NG but 보정 횟수 초과 - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            for s in (2, 3, 4):
                self._step_set_pending(s)
                
            thr_gamma = float(thr_g) if thr_g is not None else 0.05
            thr_c_val = float(thr_c) if thr_c is not None else 0.003

            self._run_batch_correction_with_jacobian(
                iter_idx=iter_idx+1,
                max_iters=max_iters,
                thr_gamma=thr_gamma,
                thr_c=thr_c_val,
                metrics=metrics
            )
            return
            
            # 무시하세요
            # if not getattr(self, "_failover_vac_applied3335", False):
            #         logging.info("[Failover] Spec NG — 대체 VAC(pk=3335) 적용 및 재평가 시도")
            #         self._failover_vac_applied3335 = True
            #         thr_gamma = float(thr_g) if thr_g is not None else 0.05
            #         thr_c_val = float(thr_c) if thr_c is not None else 0.003
            #         self._apply_vac_by_pk_and_re_evaluate(
            #             vac_info_pk=3336,
            #             thr_gamma=thr_gamma,
            #             thr_c=thr_c_val
            #         )
            #         return
            
        finally:
            self._spec_thread = None

그리고 이후, 보정 메서드가 잘 되어있는지 확인해주세요
    def _run_batch_correction_with_jacobian(self, iter_idx, max_iters, thr_gamma, thr_c, lam=1e-3, metrics=None):
        logging.info(f"[Batch Correction] iteration {iter_idx} start (Jacobian dense)")

        # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
        if not hasattr(self, "_J_dense"):
            logging.error("[Batch Correction] J_dense not loaded") # self._J_dense 없음
            return
        self._load_mapping_index_gray_to_lut()
        if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
            logging.error("[Batch Correction] no VAC cache; need latest TV VAC JSON")
            return

        # 1) NG gray 리스트 / Δ 타깃 준비
        if metrics is not None and "ng_grays" in metrics and "dG" in metrics:
            ng_list = list(metrics["ng_grays"])
            d_targets = {
                "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
                "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
                "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
            }
            thr_gamma = float(metrics.get("thr_gamma", thr_gamma))
            thr_c     = float(metrics.get("thr_c",     thr_c))
            logging.info(f"[Batch Correction] reuse metrics from SpecEvalThread, NG={ng_list}")
        else:
            dG, dCx, dCy, ng_list = SpecEvalThread.compute_gray_errors_and_ng_list(
                self._off_store, self._on_store,
                thr_gamma=thr_gamma, thr_c=thr_c
            )
            d_targets = {
                "Gamma": dG.astype(np.float32),
                "Cx":    dCx.astype(np.float32),
                "Cy":    dCy.astype(np.float32),
            }
            logging.info(f"[Batch Correction] NG grays (recomputed): {ng_list}")

        if not ng_list:
            logging.info("[Batch Correction] no NG gray (또는 0/1/254/255만 NG) → 보정 없음")
            return
    
        # 2) 현재 High LUT 확보
        vac_dict = self._vac_dict_cache

        RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
        GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
        BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

        RH = RH0.copy()
        GH = GH0.copy()
        BH = BH0.copy()

        # 3) index별 Δ 누적 (여러 gray가 같은 index를 참조할 수 있으므로)
        delta_acc = {
            "R": np.zeros_like(RH),
            "G": np.zeros_like(GH),
            "B": np.zeros_like(BH),
        }
        count_acc = {
            "R": np.zeros_like(RH, dtype=np.int32),
            "G": np.zeros_like(GH, dtype=np.int32),
            "B": np.zeros_like(BH, dtype=np.int32),
        }

        mapLUT = self._mapping_index_gray_to_lut
        
        n_gray = 256
        dR_gray = np.full(n_gray, np.nan, np.float32)
        dG_gray = np.full(n_gray, np.nan, np.float32)
        dB_gray = np.full(n_gray, np.nan, np.float32)
        corr_flag = np.zeros(n_gray, np.int32)
        
        wCx_gray = np.full(n_gray, np.nan, np.float32)
        wCy_gray = np.full(n_gray, np.nan, np.float32)
        wG_gray = np.full(n_gray, np.nan, np.float32)
        
        # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
        for g in ng_list:            
            if 0 <= g < n_gray:
                corr_flag[g] = 1
                
            dX = self._solve_delta_rgb_for_gray(
                g,
                d_targets,
                lam=lam,
                thr_c=thr_c,          # 색좌표 스펙 (예: 0.003)
                thr_gamma=thr_gamma,  # 감마 스펙 (예: 0.05)
                base_wCx=0.5,         # Cx 기본 가중치 (기존 0.5를 base로 사용)
                base_wCy=0.5,         # Cy 기본 가중치
                base_wG=1.0,          # Gamma 기본 가중치
                boost=3.0,            # NG일 때 배율
                keep=0.2,             # OK일 때 배율 (거의 무시)
            )
            if dX is None:
                continue

            dR, dG, dB, wCx_g, wCy_g, wG_g, step_gain = dX
            
            if 0 <= g < n_gray:
                dR_gray[g] = dR
                dG_gray[g] = dG
                dB_gray[g] = dB
                wCx_gray[g] = wCx_g
                wCy_gray[g] = wCy_g
                wG_gray[g] = wG_g

            idx = int(mapLUT[g])

            if 0 <= idx < len(RH):
                delta_acc["R"][idx] += dR
                count_acc["R"][idx] += 1
            if 0 <= idx < len(GH):
                delta_acc["G"][idx] += dG
                count_acc["G"][idx] += 1
            if 0 <= idx < len(BH):
                delta_acc["B"][idx] += dB
                count_acc["B"][idx] += 1

        # 5) index별 평균 Δ 적용 + clip + monotone + 로그
        for ch, arr, arr0 in (
            ("R", RH, RH0),
            ("G", GH, GH0),
            ("B", BH, BH0),
        ):
            da = delta_acc[ch]
            ct = count_acc[ch]
            mask = ct > 0

            if not np.any(mask):
                logging.info(f"[Batch Correction] channel {ch}: no indices updated")
                continue

            # 평균 Δ
            arr[mask] = arr0[mask] + (da[mask] / ct[mask])
            # clip
            arr[:] = np.clip(arr, 0.0, 4095.0)
            # 단조 증가 (i<j → LUT[i] ≤ LUT[j])
            self._enforce_monotone(arr)

        # 6) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "RchannelHigh": RH,
            "GchannelHigh": GH,
            "BchannelHigh": BH,
        }
        for k in new_lut_4096:            
            arr = np.asarray(new_lut_4096[k], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            new_lut_4096[k] = np.clip(np.round(arr), 0, 4095).astype(np.uint16)
            
        df_corr = self._build_batch_corr_df(
                iter_idx=iter_idx,
                d_targets=d_targets,
                dR_gray=dR_gray,
                dG_gray=dG_gray,
                dB_gray=dB_gray,
                corr_flag=corr_flag,
                mapLUT=mapLUT,
                RH0=RH0, GH0=GH0, BH0=BH0,
                RH=RH, GH=GH, BH=BH,
                wCx_gray=wCx_gray,
                wCy_gray=wCy_gray,
                wG_gray=wG_gray,
            )
        logging.info(
            f"[Batch Correction] {iter_idx}회차 보정 결과:\n"
            + df_corr.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )
        self._save_batch_corr_df(iter_idx, df_corr, step_gain=step_gain)
                   
        # 보정 LUT 시각화
        lut_dict_plot = {
            "R_Low":  new_lut_4096["RchannelLow"],
            "R_High": new_lut_4096["RchannelHigh"],
            "G_Low":  new_lut_4096["GchannelLow"],
            "G_High": new_lut_4096["GchannelHigh"],
            "B_Low":  new_lut_4096["BchannelLow"],
            "B_High": new_lut_4096["BchannelHigh"],
        }
        self._update_lut_chart_and_table(lut_dict_plot)

        # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
        logging.info(f"[VAC Writing] LUT {iter_idx}차 보정 VAC Data TV Writing start")

        vac_write_json = self.build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=new_lut_4096
        )
        vac_dict = json.loads(vac_write_json)
        self._vac_dict_cache = vac_dict

        def _after_write(ok, msg):
            logging.info(f"[VAC Writing] write result: {ok} {msg}")
            if not ok:
                return
            logging.info("[VAC Reading] TV reading after write")
            self._read_vac_from_tv(_after_read_back)

        def _after_read_back(vac_dict_after):
            self.send_command(self.ser_tv, 'exit')
            if not vac_dict_after:
                logging.error("[VAC Reading] TV read-back failed")
                return
            logging.info("[VAC Reading] VAC Reading 완료. Written VAC 데이터와의 일치 여부를 판단합니다.")
            mismatch_keys = self._verify_vac_data_match(written_data=vac_dict, read_data=vac_dict_after)
            if mismatch_keys:
                logging.warning("[VAC Reading] VAC 데이터 불일치 - 최적화 루프 종료")
                return
            else:
                logging.info("[VAC Reading] Written VAC 데이터와 Read VAC 데이터 일치")            
            self._step_done(3)
            
            self._fine_mode = False

            self.vac_optimization_gamma_chart.reset_on()
            self.vac_optimization_cie1976_chart.reset_on()

            profile_corr = SessionProfile(
                session_mode=f"CORR #{iter_idx}",
                cie_label=None,
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7,
                            "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )

            def _after_corr(store_corr):
                self._step_done(4)
                self._on_store = store_corr
                self._update_last_on_lv_norm(store_corr)
                
                self._step_start(5)
                self._spec_thread = SpecEvalThread(
                    self._off_store, self._on_store,
                    thr_gamma=thr_gamma, thr_c=thr_c, parent=self
                )
                self._spec_thread.finished.connect(
                    lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx, max_iters)
                )
                self._spec_thread.start()

            logging.info(f"[Measurement] LUT {iter_idx}차 보정 기준 re-measure start (after LUT update)")
            self._step_start(4)
            self.start_viewing_angle_session(
                profile=profile_corr,
                gray_levels=op.gray_levels_256,
                gamma_patterns=('white',),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000,
                gamma_settle_ms=1000,
                cs_settle_ms=1000,
                on_done=_after_corr
            )

        self._step_start(3)
        self._write_vac_to_tv(vac_write_json, on_finished=_after_write)

    def _solve_delta_rgb_for_gray(
        self,
        g: int,
        d_targets: dict,
        lam: float = 1e-3,
        # --- (옵션1) 기존처럼 직접 weight 지정하고 싶을 때 ---
        wCx: float | None = None,
        wCy: float | None = None,
        wG:  float | None = None,
        # --- (옵션2) NG 정도에 따라 자동 가중치 계산 ---
        thr_c: float | None = None,
        thr_gamma: float | None = None,
        base_wCx: float = 1.0,
        base_wCy: float = 1.0,
        base_wG:  float = 1.0,
        boost: float = 3.0,
        keep: float = 0.2,
    ):
        """
        주어진 gray g에서, 현재 ΔY = [dCx, dCy, dGamma]를
        자코비안 J_g를 이용해 줄이기 위한 ΔX = [ΔR_H, ΔG_H, ΔB_H]를 푼다.

        관계식:  ΔY_new ≈ ΔY + J_g · ΔX
        우리가 원하는 건 ΔY_new ≈ 0 이므로, J_g · ΔX ≈ -ΔY 를 풀어야 함.

        리지 가중 최소자승:
            argmin_ΔX || W (J_g ΔX + ΔY) ||^2 + λ ||ΔX||^2
            → (J^T W^2 J + λI) ΔX = - J^T W^2 ΔY

        - thr_c, thr_gamma가 주어지면:
            NG 여부에 따라 (base_w * boost) / (base_w * keep)로 가중치 자동 계산
        - thr_c, thr_gamma가 None 이고 wCx/wCy/wG가 주어지면:
            예전 방식처럼 고정 weight 사용
        """
        Jg = np.asarray(self._J_dense[g], dtype=np.float32)  # (3,3)
        if not np.isfinite(Jg).all():
            logging.warning(f"[BATCH CORR] g={g}: J_g has NaN/inf → skip")
            return None

        dCx_g = float(d_targets["Cx"][g])
        dCy_g = float(d_targets["Cy"][g])
        dG_g  = float(d_targets["Gamma"][g])
        dy = np.array([dCx_g, dCy_g, dG_g], dtype=np.float32)  # (3,)

        # target이 NaN/Inf인 경우
        if not np.isfinite(dy).all():
            logging.warning(
                f"[BATCH CORR] g={g}: dY has NaN/inf "
                f"(dCx, dCy, dG) = ({dCx_g}, {dCy_g}, {dG_g}) → skip this gray"
            )
            return None
        
        # 이미 거의 0이면 굳이 보정 안 해도 됨
        if np.all(np.abs(dy) < 1e-6):
            return None

        # ---------------------------------------------
        # 1) 가중치 계산
        #    - 우선순위:
        #      (1) thr_c/thr_gamma가 있으면 NG 기반 자동 가중치
        #      (2) 아니면 (wCx,wCy,wG) 직접 지정값 사용
        #      (3) 둘 다 없으면 base_w* 그대로 사용
        # ---------------------------------------------
        if thr_c is not None and thr_gamma is not None:
            def w_for(err: float, thr: float, base: float) -> float:
                ratio = abs(err) / max(thr, 1e-6)
                ratio_clamped = min(ratio, 1.0)
                w = base * (keep) + (boost - keep) * ratio_clamped
                return w

            wCx_eff = w_for(dCx_g, thr_c, base_wCx)
            wCy_eff = w_for(dCy_g, thr_c, base_wCy)
            wG_eff  = w_for(dG_g,  thr_gamma, base_wG)

        elif (wCx is not None) and (wCy is not None) and (wG is not None):
            # 옛날 방식: 직접 weight 지정
            wCx_eff, wCy_eff, wG_eff = float(wCx), float(wCy), float(wG)

        else:
            # fallback: 그냥 base weight 사용
            wCx_eff, wCy_eff, wG_eff = base_wCx, base_wCy, base_wG

        w_vec = np.array([wCx_eff, wCy_eff, wG_eff], dtype=np.float32)

        # ---------------------------------------------
        # 2) 가중 least squares (기존 로직 그대로)
        # ---------------------------------------------
        WJ = w_vec[:, None] * Jg   # (3,3)
        Wy = w_vec * dy            # (3,)

        A = WJ.T @ WJ + float(lam) * np.eye(3, dtype=np.float32)  # (3,3)
        b = - WJ.T @ Wy                                           # (3,)

        try:
            dX = np.linalg.solve(A, b).astype(np.float32)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

        step_gain = 1.0
        dR, dG, dB = (float(dX[0]) * step_gain,
                    float(dX[1]) * step_gain,
                    float(dX[2]) * step_gain)

        return dR, dG, dB, wCx_eff, wCy_eff, wG_eff, step_gain

