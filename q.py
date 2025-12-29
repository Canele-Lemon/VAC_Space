if spec_ok:
    self._step_done(5)
    logging.info("[Evaluation] Spec 통과 — 최적화 종료")

    # ✅ 다운로드용 최종 VAC 텍스트(포맷 유지) 저장
    try:
        self._final_vac_text_for_download = self.build_vacparam_std_format(
            base_vac_dict=self._vac_dict_cache,
            new_lut_tvkeys=None
        )
    except Exception:
        logging.exception("[Download] final vac text build failed")
        self._final_vac_text_for_download = None

    self.ui.vac_btn_JSONdownload.setEnabled(True)
    return
    
import os
import sys
import tempfile
import subprocess
from datetime import datetime

def _on_click_download_vac(self):
    """
    spec 통과로 확정된 VAC 데이터를
    build_vacparam_std_format() 포맷 그대로 임시파일로 저장 후 텍스트 뷰어로 오픈.
    """
    try:
        vac_text = getattr(self, "_final_vac_text_for_download", None)

        # 혹시 저장 안 돼있으면, 캐시 dict로라도 재조립 시도
        if not vac_text:
            if getattr(self, "_vac_dict_cache", None) is None:
                logging.error("[Download] no final VAC text and no vac cache.")
                return
            vac_text = self.build_vacparam_std_format(
                base_vac_dict=self._vac_dict_cache,
                new_lut_tvkeys=None
            )

        # 파일명: VAC_YYYYmmdd_HHMMSS.json (내용은 json-like지만 포맷 커스텀)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"VAC_{ts}.json"

        # 임시폴더에 파일 생성 (삭제하지 않음: 사용자 확인/복사 편의)
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, fname)

        # ✅ 포맷 유지 위해 json dump 금지. 그냥 raw text로 저장.
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(vac_text)

        logging.info(f"[Download] VAC temp file saved: {path}")

        # --- OS별로 텍스트 뷰어 열기 ---
        if sys.platform.startswith("win"):
            # Windows: Notepad로 열기 (가장 확실)
            subprocess.Popen(["notepad.exe", path], close_fds=True)
        elif sys.platform == "darwin":
            # macOS: 기본 앱
            subprocess.Popen(["open", path], close_fds=True)
        else:
            # Linux: 기본 앱
            subprocess.Popen(["xdg-open", path], close_fds=True)

    except Exception:
        logging.exception("[Download] open temp viewer failed")