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

        # 5) 평가 정책 마스크
        # - 0,1: 전부 제외
        # - 2~5: Gamma-only
        # - 248~255: Color-only
        # - 6~247: 둘 다 평가
        exclude_all     = {0, 1}
        gamma_only_set  = set(range(2, 6))          # 2..5
        color_only_set  = set(range(248, 256))      # 248..255

        gamma_eval_grays = set(range(2, 248))       # 2..247 (248..255는 Gamma 제외)
        color_eval_grays = set(range(6, 256))       # 6..255 (2..5는 색좌표 제외)

        # exclude_all은 두 마스크에서 모두 제거
        gamma_eval_grays -= exclude_all
        color_eval_grays -= exclude_all

        # 6) NG 리스트 계산 (마스크 적용)
        ng_grays_gamma = [g for g in sorted(gamma_eval_grays)
                          if np.isfinite(abs_dG[g]) and (abs_dG[g] > thr_gamma)]

        ng_grays_color = [g for g in sorted(color_eval_grays)
                          if (np.isfinite(abs_dCx[g]) and abs_dCx[g] > thr_c) or
                             (np.isfinite(abs_dCy[g]) and abs_dCy[g] > thr_c)]

        # 전체 NG (중복 제거)
        ng_grays = sorted(set(ng_grays_gamma) | set(ng_grays_color))


        return dG, dCx, dCy, ng_grays, ng_grays_gamma, ng_grays_color, gamma_eval_grays, color_eval_grays

    def run(self):
        try:
            dG, dCx, dCy, ng_grays, ng_gamma, ng_color, gamma_eval_grays, color_eval_grays = \
                            self.compute_gray_errors_and_ng_list(
                                self.off_store, self.on_store,
                                self.thr_gamma, self.thr_c
                            )
                            
            abs_dG  = np.abs(dG)
            abs_dCx = np.abs(dCx)
            abs_dCy = np.abs(dCy)
            
            if gamma_eval_grays:
                max_dG = float(np.nanmax(np.array([abs_dG[g]  for g in gamma_eval_grays], dtype=np.float64)))
            else:
                max_dG = float('nan')

            if color_eval_grays:
                max_dCx = float(np.nanmax(np.array([abs_dCx[g] for g in color_eval_grays], dtype=np.float64)))
                max_dCy = float(np.nanmax(np.array([abs_dCy[g] for g in color_eval_grays], dtype=np.float64)))
            else:
                max_dCx = float('nan')
                max_dCy = float('nan')

            spec_ok = (len(ng_grays) == 0)

            metrics = {
                "max_dG":  max_dG,
                "max_dCx": max_dCx,
                "max_dCy": max_dCy,
                "thr_gamma": self.thr_gamma,
                "thr_c": self.thr_c,

                "dG":  dG,
                "dCx": dCx,
                "dCy": dCy,

                # NG/마스크를 분리해 전달 (로깅/후속 로직에 유용)
                "ng_grays": ng_grays,
                "ng_grays_gamma": ng_gamma,
                "ng_grays_color": ng_color,
                "gamma_eval_grays": sorted(gamma_eval_grays),
                "color_eval_grays": sorted(color_eval_grays),
            }
            self.finished.emit(spec_ok, metrics)

        except Exception:
            self.finished.emit(False, {"error": True})
