def _after_read_back(vac_dict_after):
    if not vac_dict_after:
        logging.error("보정 후 VAC 재읽기 실패")
        return

    # 1) 캐시/차트 갱신
    self._vac_dict_cache = vac_dict_after
    lut_dict_plot = {k.replace("channel","_"): v
                     for k, v in vac_dict_after.items() if "channel" in k}
    self._update_lut_chart_and_table(lut_dict_plot)  # 내부에서 LUTChart.reset_and_plot 호출

    # 2) ON 시리즈 리셋 (OFF는 참조 유지)
    self.vac_optimization_gamma_chart.reset_on()
    self.vac_optimization_cie1976_chart.reset_on()

    # 3) 보정 후(=ON) 측정 세션 시작
    profile_corr = SessionProfile(
        legend_text=f"CORR #{iter_idx}",   # state 판정은 'VAC OFF' prefix 여부로 하므로 여기선 ON 상태로 처리됨
        cie_label=None,                    # data_1/2 안 씀
        table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
        ref_store=self._off_store          # 항상 OFF 대비 Δ를 계산
    )

    def _after_corr(store_corr):
        self._on_store = store_corr  # 최신 ON(보정 후) 측정 결과로 교체
        if self._check_spec_pass(self._off_store, self._on_store):
            logging.info("✅ 스펙 통과 — 최적화 종료")
            return
        if iter_idx < max_iters:
            logging.info(f"🔁 스펙 out — 다음 보정 사이클로 진행 (iter={iter_idx+1})")
            self._run_correction_iteration(iter_idx=iter_idx+1, max_iters=max_iters)
        else:
            logging.info("⛔ 최대 보정 횟수 도달 — 종료")

    self.start_viewing_angle_session(
        profile=profile_corr,
        gray_levels=getattr(op, "gray_levels_256", list(range(256))),
        gamma_patterns=('white',),                 # ✅ 감마는 white만 측정
        colorshift_patterns=op.colorshift_patterns,
        first_gray_delay_ms=3000,
        cs_settle_ms=1000,
        on_done=_after_corr
    )