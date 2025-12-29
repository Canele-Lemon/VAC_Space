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

                # ✅ 이제 thr는 policy에서만 온다
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