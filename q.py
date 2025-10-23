import re

def _get_ui_meta(self):
    """
    UI 콤보값에서 패널명, 프레임레이트(Hz 제거), 모델연도(Y 제거)를 파싱해 반환.
    실패 시 0.0으로 폴백하며 로그 남김.
    """
    # Panel text
    panel_text = self.ui.vac_cmb_PanelMaker.currentText().strip()

    # Frame rate: "120Hz", "119.88 Hz" 등 → 숫자만 추출
    fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
    fr_num = 0.0
    try:
        m = re.search(r'(\d+(?:\.\d+)?)', fr_text)
        if m:
            fr_num = float(m.group(1))
        else:
            logging.warning(f"[UI META] FrameRate parsing failed: '{fr_text}' → 0.0")
    except Exception as e:
        logging.warning(f"[UI META] FrameRate parsing error for '{fr_text}': {e}")

    # Model year: "Y2024" / "2024Y" / "2024" → 숫자만 추출
    my_text = self.ui.vac_cmb_ModelYear.currentText().strip() if hasattr(self.ui, "vac_cmb_ModelYear") else ""
    model_year = 0.0
    try:
        m = re.search(r'(\d{2,4})', my_text)  # 23, 2023 등
        if m:
            model_year = float(m.group(1))
        else:
            logging.warning(f"[UI META] ModelYear parsing failed: '{my_text}' → 0.0")
    except Exception as e:
        logging.warning(f"[UI META] ModelYear parsing error for '{my_text}': {e}")

    return panel_text, fr_num, model_year