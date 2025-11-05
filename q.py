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

여기서
    def _set_vac_active(self, enable: bool) -> bool:
        try:
            logging.debug("[TV Control] 현재 VAC 적용 상태를 확인합니다.")
            current_status = self._check_vac_status()
            current_active = bool(current_status.get("activated", False))

            if current_active == enable:
                logging.info(f"[TV Control] VAC already {'ON' if enable else 'OFF'} - skipping command.")
                return True

            self.send_command(self.ser_tv, 's')
            setVACActive = (
                "luna-send -n 1 -f "
                "luna://com.webos.service.panelcontroller/setVACActive "
                f"'{{\"OnOff\":{str(enable).lower()}}}'"
            )
            self.send_command(self.ser_tv, setVACActive)
            self.send_command(self.ser_tv, 'exit', output_limit=22)
            time.sleep(0.5)
            st = self._check_vac_status()
            return bool(st.get("activated", False)) == enable
        
        except Exception as e:
            logging.error(f"[TV Control] VAC {'ON' if enable else 'OFF'} 전환 실패: {e}")
            return False
        
    def _check_vac_status(self):
        self.send_command(self.ser_tv, 's')
        getVACSupportstatus = 'luna-send -n 1 -f luna://com.webos.service.panelcontroller/getVACSupportStatus \'{"subscribe":true}\''
        VAC_support_status = self.send_command(self.ser_tv, getVACSupportstatus)
        VAC_support_status = self.extract_json_from_luna_send(VAC_support_status)
        self.send_command(self.ser_tv, 'exit', output_limit=22)
        
        if not VAC_support_status:
            logging.warning("[TV Control] Failed to retrieve VAC support status from TV.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        if not VAC_support_status.get("isSupport", False):
            logging.info("[TV Control] VAC is not supported on this model.")
            return {"supported": False, "activated": False, "vacdata": None}
        
        activated = VAC_support_status.get("isActivated", False)
        logging.info(f"[TV Control] VAC 적용 상태: {activated}")
                
        return {"supported": True, "activated": activated}
를 통해 VAC를 끄는 방식 말고, 
    def _fetch_vac_by_vac_info_pk(self, pk: int):
        """
        `W_VAC_Info` 테이블에서 주어진 `PK` 값으로 `VAC_Version`과 `VAC_Data`를 가져옵니다.
        반환: (vac_version, vac_data) 또는 (None, None)
        """
        try:
            db_conn = pymysql.connect(**config.conn_params)
            cursor = db_conn.cursor()

            cursor.execute("""
                SELECT `VAC_Version`, `VAC_Data`
                FROM `W_VAC_Info`
                WHERE `PK` = %s
            """, (pk,))

            vac_row = cursor.fetchone()

            if not vac_row:
                logging.error(f"[DB] No VAC information found for PK={pk}")
                return None, None

            vac_version = vac_row[0]
            vac_data = vac_row[1]

            logging.info(f"[DB] VAC Info fetched for PK={pk} - Version: {vac_version}")
            return vac_version, vac_data

        except Exception as e:
            logging.error(f"[DB] Error while fetching VAC Info by PK={pk}: {e}")
            return None, None
를 통해 PK=1인 것을 tv writing하는 식으로 바꿀래요

마찬가지로 아래 _run_off_baseline_then_on 메서드에도 vac를 키는 것은 따로 안하겠습니다.

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
            self._gamma_off_vec = self._compute_gamma_series(lv_off)
            
            self._step_done(1)
            logging.info("[Measurement] VAC OFF 상태 측정 완료")
            
            logging.info("[TV Control] VAC ON 전환 시작")
            if not self._set_vac_active(True):
                logging.warning("[TV Control] VAC ON 전환 실패 - VAC 최적화 종료")
                return
            logging.info("[TV Control] VAC ON 전환 성공")
            
            logging.info("[Measurement] VAC ON 측정 시작")
            self._apply_vac_from_db_and_measure_on()

        self.start_viewing_angle_session(
            profile=profile_off,
            gray_levels=op.gray_levels_256,
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            cs_settle_ms=1000,
            on_done=_after_off
        )
