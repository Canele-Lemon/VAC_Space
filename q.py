from PySide2.QtCore import QThread, Signal
import numpy as np

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

위 스레스 클래스에서 NG gray 리스트 뽑기를 하기 위해 코드를 수정하는게 좋을까요 아니면, 앞서 알려주신 NG gray 리스트만 뽑는 helper를 새로 만드는게 좋을까요?
