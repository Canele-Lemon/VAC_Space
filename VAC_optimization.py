# 최상단 import에 추가
from PySide2.QtCore import QThread, Signal

class SpecEvalThread(QThread):
    finished = Signal(bool, dict)  # (spec_ok, metrics)

    def __init__(self, off_store, on_store, thr_gamma=0.05, thr_c=0.003, parent=None):
        super().__init__(parent)
        self.off_store = off_store
        self.on_store  = on_store
        self.thr_gamma = float(thr_gamma)
        self.thr_c     = float(thr_c)

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

    def run(self):
        try:
            lv_off, cx_off, cy_off = self._extract_white(self.off_store)
            lv_on , cx_on , cy_on  = self._extract_white(self.on_store)

            G_off = self._compute_gamma_series(lv_off)
            G_on  = self._compute_gamma_series(lv_on)

            dG  = np.abs(G_on - G_off)
            dCx = np.abs(cx_on - cx_off)
            dCy = np.abs(cy_on - cy_off)

            max_dG  = float(np.nanmax(dG))
            max_dCx = float(np.nanmax(dCx))
            max_dCy = float(np.nanmax(dCy))

            spec_ok = (max_dG <= self.thr_gamma) and (max_dCx <= self.thr_c) and (max_dCy <= self.thr_c)
            metrics = {"max_dG": max_dG, "max_dCx": max_dCx, "max_dCy": max_dCy,
                       "thr_gamma": self.thr_gamma, "thr_c": self.thr_c}
            self.finished.emit(spec_ok, metrics)
        except Exception:
            self.finished.emit(False, {"error": True})
            
def _after_corr(store_corr):
    self.stop_loading_animation(self.label_processing_step_4, self.movie_processing_step_4)
    self._on_store = store_corr

    # 스펙 계산 스레드 시작 (PySide2 Signal)
    self._spec_thread = SpecEvalThread(self._off_store, self._on_store, thr_gamma=0.05, thr_c=0.003, parent=self)
    self._spec_thread.finished.connect(lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx, max_iters))
    self._spec_thread.start()
    
def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
    try:
        if metrics and "error" not in metrics:
            logging.info(
                f"[SPEC(thread)] max|ΔGamma|={metrics['max_dG']:.6f} (≤{metrics['thr_gamma']}), "
                f"max|ΔCx|={metrics['max_dCx']:.6f}, max|ΔCy|={metrics['max_dCy']:.6f} (≤{metrics['thr_c']})"
            )
        else:
            logging.warning("[SPEC(thread)] evaluation failed — treating as not passed.")

        # UI 표/차트 갱신은 메인스레드에서
        self._update_spec_views(self._off_store, self._on_store)

        if spec_ok:
            logging.info("✅ 스펙 통과 — 최적화 종료")
            return
        if iter_idx < max_iters:
            logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
            self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
        else:
            logging.info("⛔ 최대 보정 횟수 도달 — 종료")
    finally:
        self._spec_thread = None
        
        