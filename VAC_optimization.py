def _get_ui_meta(self):
    """
    UI 콤보에서 panel / frame_rate / model_year(두 자리 float)를 가져온다.
    - FrameRate: "60Hz", "119.88 Hz" 등에서 숫자만 추출
    - ModelYear: 기본 형식 "Y26" ← 권장
      (안전장치로 "26Y"도 허용하되, 우선순위는 "Y{2자리}" 매칭)
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

    # FrameRate
    try:
        fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
        m = re.search(r'(\d+(?:\.\d+)?)', fr_text)  # 숫자만
        fr_val = float(m.group(1)) if m else 0.0
    except Exception as e:
        logging.debug(f"[UI META] FrameRate 파싱 에러: {e}")

    # ModelYear (우선: 'Y26' 정확 매칭 → 폴백: '26Y')
    try:
        if hasattr(self, "ui") and hasattr(self.ui, "vac_cmb_ModelYear"):
            my_text = self.ui.vac_cmb_ModelYear.currentText().strip()
            # 우선순위 1: 'Y' + 2자리
            m1 = re.match(r'^[Yy]\s*(\d{2})$', my_text)
            if m1:
                my_val = float(m1.group(1))
            else:
                # 우선순위 2: 2자리 + 'Y'
                m2 = re.match(r'^(\d{2})\s*[Yy]$', my_text)
                if m2:
                    my_val = float(m2.group(1))
                else:
                    logging.debug(f"[UI META] ModelYear 형식 비정상: '{my_text}' → 0.0")
        else:
            logging.debug("[UI META] ModelYear 콤보 없음 → 0.0")
    except Exception as e:
        logging.debug(f"[UI META] ModelYear 파싱 에러: {e}")

    logging.debug(f"[UI META] panel='{panel_text}', fr='{fr_val}Hz', model_year='Y{int(my_val):02d}'")
    return panel_text, fr_val, my_val