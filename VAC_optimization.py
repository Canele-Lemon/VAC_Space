def _get_ui_meta(self):
    """
    UI 콤보에서 패널명/프레임레이트/모델연도(두 자리 숫자+Y)만 간단 추출해서 반환.
    - panel_text: 그대로
    - frame_rate: "60Hz", "119.88 Hz" 등에서 숫자만 float
    - model_year: "25Y" 형태에서 앞 숫자만 float(예: 25.0)
    실패/예외 시 0.0으로 폴백.
    """
    import re
    panel_text = ""
    fr_val = 0.0
    my_val = 0.0

    # Panel
    try:
        panel_text = self.ui.vac_cmb_PanelMaker.currentText().strip()
    except Exception as e:
        logging.debug(f"[UI META] Panel text 읽기 실패: {e}")

    # Frame rate: "...Hz" -> 숫자만
    try:
        fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
        m = re.search(r'(\d+(?:\.\d+)?)', fr_text)
        if m:
            fr_val = float(m.group(1))
        else:
            logging.debug(f"[UI META] FrameRate 형식 비정상: '{fr_text}' → 0.0")
    except Exception as e:
        logging.debug(f"[UI META] FrameRate 파싱 에러: {e}")

    # Model year: "25Y" 고정 전제 → 숫자만
    try:
        if hasattr(self.ui, "vac_cmb_ModelYear"):
            my_text = self.ui.vac_cmb_ModelYear.currentText().strip()
            m = re.match(r'^\s*(\d{1,4})\s*[Yy]\s*$', my_text)  # "25Y" or "2025Y"도 허용
            if m:
                my_val = float(m.group(1))  # 전제상 25 → 25.0
            else:
                logging.debug(f"[UI META] ModelYear 형식 비정상: '{my_text}' → 0.0")
        else:
            logging.debug("[UI META] ModelYear 콤보 없음 → 0.0")
    except Exception as e:
        logging.debug(f"[UI META] ModelYear 파싱 에러: {e}")

    logging.debug(f"[UI META] panel='{panel_text}', fr='{fr_val}Hz', model_year='{my_val}Y'")
    return panel_text, fr_val, my_val