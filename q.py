    def _start_fine_correction_for_ng_list(self, ng_grays, thr_gamma=0.05, thr_c=0.003):
        # unique + 정렬
        ng_sorted = sorted({int(g) for g in ng_grays})
        if not ng_sorted:
            logging.info("[FINE] NG gray list empty → nothing to do")
            return

        logging.info(f"[FINE] start fine correction session for NG grays: {ng_sorted}")

        # fine 모드 ON
        self._fine_mode = True
        self._fine_ng_list = ng_sorted

        # ON 차트 초기화 (원하면 유지해도 됨)
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        profile_fine = SessionProfile(
            legend_text="CORR_FINE",
            cie_label=None,
            table_cols={
                "lv":4, "cx":5, "cy":6, "gamma":7,
                "d_cx":8, "d_cy":9, "d_gamma":10
            },
            ref_store=self._off_store
        )

        def _after_fine(store_corr):
            # fine 세션에서 만들어진 ON 데이터를 on_store로 저장
            self._step_done(4)
            self._on_store = store_corr

            # fine 모드 끝 (이후 세션은 per-gray 자동보정 안 함)
            self._fine_mode = False

            # 최종 Spec 평가 한 번 더 (추가 보정은 하지 않기 위해 max_iters=0)
            self._step_start(5)
            self._spec_thread = SpecEvalThread(
                self._off_store,
                self._on_store,
                thr_gamma=thr_gamma,
                thr_c=thr_c,
                parent=self
            )
            self._spec_thread.finished.connect(
                # max_iters=0 → _on_spec_eval_done 안에서 추가 보정 루프 없음
                lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx=0, max_iters=0)
            )
            self._spec_thread.start()

        self._step_start(4)
        self.start_viewing_angle_session(
            profile=profile_fine,
            gray_levels=ng_sorted,          # NG gray만 측정
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            gamma_settle_ms = 1000,
            cs_settle_ms=1000,
            on_done=_after_fine
        )
    def _session_step(self):
        s = self._sess
        if s.get('paused', False):
            return
        
        if s['phase'] == 'gamma':
            if s['p_idx'] >= len(s['patterns']):
                s['phase'] = 'colorshift'
                s['cs_idx'] = 0
                QTimer.singleShot(60, lambda: self._session_step())
                return

            if s['g_idx'] >= len(s['gray_levels']):
                s['g_idx'] = 0
                s['p_idx'] += 1
                QTimer.singleShot(40, lambda: self._session_step())
                return

            pattern = s['patterns'][s['p_idx']]
            gray = s['gray_levels'][s['g_idx']]

            if pattern == 'white':
                rgb_value = f"{gray},{gray},{gray}"
            elif pattern == 'red':
                rgb_value = f"{gray},0,0"
            elif pattern == 'green':
                rgb_value = f"0,{gray},0"
            else:
                rgb_value = f"0,0,{gray}"
            self.changeColor(rgb_value)

            if s['g_idx'] == 0:
                delay = s['first_gray_delay_ms']
            else:
                delay = s.get('gamma_settle_ms', 0)
            QTimer.singleShot(delay, lambda p=pattern, g=gray: self._trigger_gamma_pair(p, g))

        elif s['phase'] == 'colorshift':
            if s['cs_idx'] >= len(s['cs_patterns']):
                s['phase'] = 'done'
                QTimer.singleShot(0, lambda: self._session_step())
                return

            pname, r, g, b = s['cs_patterns'][s['cs_idx']]
            self.changeColor(f"{r},{g},{b}")
            QTimer.singleShot(s['cs_settle_ms'], lambda pn=pname: self._trigger_colorshift_pair(pn))

        else:  # done
            self._finalize_session()

이렇게 되어 있을 때
NG gray만 측정할 때 패턴 변경 -> 측정 사이 대기 시간이 1000으로 설정되어있다는 말씀이시죠?
