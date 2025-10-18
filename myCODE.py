    def _apply_vac_from_db_and_measure_on(self):
        # 3-a) DB에서 Panel_Maker + Frame_Rate 조합인 VAC_Data 가져오기
        panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
        fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
        vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
        if vac_data is None:
            logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다.")
            return
        
        def _after_write(ok, msg):
            logging.info(f"[VAC Write] {msg}")
            if not ok:
                return
            self._read_vac_from_tv(lambda vac_dict: _after_read(vac_dict)) # _read_vac_from_tv 메서드 끝난 후 _after_read 함수 실행
        
        def _after_read(vac_dict):
            if not vac_dict:
                logging.error("VAC 데이터 읽기 실패: 빈 dict")
                return
            
            self._vac_dict_cache = vac_dict

            vac_lut_dict = {key.replace("channel", "_"): v
                        for key, v in vac_dict.items()
                        if "channel" in key}
            self._update_lut_chart_and_table(vac_lut_dict)
            
            # VAC ON 측정 세션
            gamma_lines_on = {
                'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"OPT - {p}") 
                        for p in ('white','red','green','blue')},
                'sub':  {p: self.vac_optimization_gamma_chart.add_series(axis_index=1, label=f"OPT - {p}") 
                        for p in ('white','red','green','blue')},
            }
            profile_on = SessionProfile(
                legend_text="VAC ON",
                cie_label="data_2",
                table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
                ref_store=self._off_store
            )
            self.start_viewing_angle_session(
                profile=profile_on, gamma_lines=gamma_lines_on,
                gray_levels=op.gray_levels_256, patterns=('white','red','green','blue'),
                colorshift_patterns=op.colorshift_patterns,
                first_gray_delay_ms=3000, cs_settle_ms=1000,
                on_done=_after_on
            )
            
        def _after_on(store_on):
            self._on_store = store_on
            if self._check_spec_pass(self._off_store, self._on_store):
                logging.info("축하합니다! 스펙 통과 — 종료")
                return
            # (D) 보정 반복 시작
            self._run_correction_iteration(iter_idx=1)
                        
        # (B) VAC Data TV에 적용 → 읽기
        self._write_vac_to_tv(vac_data, on_finished=_after_write)
