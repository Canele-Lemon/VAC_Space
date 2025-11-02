def _finish_gray_fix(self, gray:int, *, pass_now: bool):
    ctx = self._sess.get('_gray_fix', None)
    if not ctx:
        self._resume_session(); return
    if pass_now or ctx['tries'] >= ctx['max']:
        logging.info(f"[GRAY-FIX] g={gray} {'PASS' if pass_now else 'MAX RETRIES'} → resume")
        self._sess['_gray_fix'] = None
        self._resume_session()
    else:
        self._do_gray_fix_once()  # 다음 재시도

def _remeasure_same_gray(self, gray:int):
    """paused 상태에서 같은 g만 다시 측정 → store 반영 → 그 자리에서 PASS 판정"""
    s = self._sess
    self.changeColor(f"{gray},{gray},{gray}")
    payload = {}

    def handle(role, res):
        payload[role] = res
        got_main = ('main' in payload)
        got_sub  = ('sub' in payload) or (self.sub_instrument_cls is None)
        if got_main and got_sub:
            # 기존 소비 로직 재사용(차트/테이블 업데이트)
            self._consume_gamma_pair('white', gray, payload)
            ok = self._is_gray_spec_ok(gray, off_store=self._off_store, on_store=s['store'])
            self._finish_gray_fix(gray, pass_now=ok)

    if self.main_instrument_cls:
        t1 = MeasureThread(self.main_instrument_cls, 'main')
        t1.measure_completed.connect(handle); t1.start()
    if self.sub_instrument_cls:
        t2 = MeasureThread(self.sub_instrument_cls, 'sub')
        t2.measure_completed.connect(handle); t2.start()