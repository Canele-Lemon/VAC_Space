    def start_VAC_optimization(self):
        """
        ============================== 메인 엔트리: 버튼 이벤트 연결용 ==============================
        전체 Flow:
        """
        for s in (1,2,3,4,5):
            self._step_set_pending(s)
        self._step_start(1)
        self._fine_mode = False
        self._fine_ng_list = None
        
        try:
            self._load_jacobian_bundle_npy()
        except Exception as e:
            logging.exception("[Jacobian] Jacobian load failed")
            return
        
        logging.info("[TV Control] VAC OFF 전환 시작")
        if not self._set_vac_active(False):
            logging.error("[TV Control] VAC OFF 전환 실패 - VAC 최적화를 종료합니다.")
            return
        logging.info("[TV Control] TV VAC OFF 전환 성공")    
        
        logging.info("[Measurement] VAC OFF 상태 측정 시작")
        self._run_off_baseline_then_on()

    def _run_off_baseline_then_on(self):
        profile_off = SessionProfile(
            legend_text="VAC OFF (Ref.)",
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
                
            self._off_lv_vec = lv_off
            self._off_lv0 = float(lv_off[0])
            with np.errstate(invalid='ignore'):
                self._off_denom = float(np.nanmax(lv_off[1:] - self._off_lv0)) if np.isfinite(self._off_lv0) else np.nan
            self._gamma_off_vec = self._compute_gamma_series(lv_off)

            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
            logging.info("[TV Control] VAC ON 전환 성공")
                
            # self._apply_vac_from_db_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off, 
            gray_levels=op.gray_levels_256, 
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            cs_settle_ms=1000,
            on_done=_after_off
        )

여기서 self._off_denom을 저장하는 이유가 있나요? off lv 최대값은 per-gray 보정때 normalized를 통한 gamma 계산을 위해 필요한 것은 이해했지만 self._off_denom를 따로 저장하는 이유는 모르겠어요
