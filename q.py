# 클래스 내부 어딘가에 추가
def _pause_session(self, reason:str=""):
    s = self._sess
    s['paused'] = True
    logging.info(f"[SESSION] paused. reason={reason}")

def _resume_session(self):
    s = self._sess
    if s.get('paused', False):
        s['paused'] = False
        logging.info("[SESSION] resumed")
        QTimer.singleShot(0, lambda: self._session_step())