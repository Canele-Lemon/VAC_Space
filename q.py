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

    # 🔹 새 helper: per-gray error + NG 리스트 계산
    @staticmethod
    def compute_gray_errors_and_ng_list(off_store, on_store, thr_gamma, thr_c):
        # 1) OFF/ON 시리즈
        lv_off, cx_off, cy_off = SpecEvalThread._extract_white(off_store)
        lv_on , cx_on , cy_on  = SpecEvalThread._extract_white(on_store)

        # 2) Gamma series
        G_off = SpecEvalThread._compute_gamma_series(lv_off)
        G_on  = SpecEvalThread._compute_gamma_series(lv_on)

        # 3) Δ = ON - OFF
        dG  = G_on  - G_off
        dCx = cx_on - cx_off
        dCy = cy_on - cy_off

        # 4) 절대값
        abs_dG  = np.abs(dG)
        abs_dCx = np.abs(dCx)
        abs_dCy = np.abs(dCy)

        # 5) NG 마스크 (여기서 edge gray는 무시)
        mask_ng = (
            (abs_dG  > thr_gamma) |
            (abs_dCx > thr_c)     |
            (abs_dCy > thr_c)
        )

        # gray 0,1,254,255는 NG여도 무시
        for e in (0, 1, 254, 255):
            if 0 <= e < 256:
                mask_ng[e] = False

        ng_grays = np.where(mask_ng)[0].astype(int).tolist()

        return dG, dCx, dCy, ng_grays

    def run(self):
        try:
            # 🔸 공통 helper 호출
            dG, dCx, dCy, ng_grays = self.compute_gray_errors_and_ng_list(
                self.off_store, self.on_store,
                self.thr_gamma, self.thr_c
            )

            abs_dG  = np.abs(dG)
            abs_dCx = np.abs(dCx)
            abs_dCy = np.abs(dCy)

            max_dG  = float(np.nanmax(abs_dG))
            max_dCx = float(np.nanmax(abs_dCx))
            max_dCy = float(np.nanmax(abs_dCy))

            # spec_ok는 "NG gray가 하나도 없다"로 정의하면 직관적
            spec_ok = (len(ng_grays) == 0)

            metrics = {
                "max_dG":  max_dG,
                "max_dCx": max_dCx,
                "max_dCy": max_dCy,
                "thr_gamma": self.thr_gamma,
                "thr_c": self.thr_c,

                # 🔸 per-gray 정보도 같이 넘겨두면 나중에 그래프/로그에 활용 가능
                "dG":  dG,
                "dCx": dCx,
                "dCy": dCy,

                # 🔸 NG gray 리스트 (자코비안 보정에서 핵심)
                "ng_grays": ng_grays,
            }
            self.finished.emit(spec_ok, metrics)

        except Exception:
            self.finished.emit(False, {"error": True})