def _on_vac_btn_computeY_clicked(self):
    """
    vac_btn_computeY 클릭 시:
    - dL_R / dL_G / dL_B(=ΔR/ΔG/ΔB) 읽어서
    - J_g · ΔX로 ΔCx / ΔCy / ΔGamma 예측
    - 결과를 lineEdit_dCx / dCy / dGamma에 표시
    """
    # 1) ΔX 입력값 읽기 (빈 문자열/잘못된 값은 0으로 처리)
    def _read_float_safe(line_edit):
        text = line_edit.text().strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            logging.warning(f"[VAC] invalid float in {line_edit.objectName()}: {text!r} → 0.0 사용")
            return 0.0

    dR = _read_float_safe(self.ui.vac_lineEdit_dL_R)
    dG = _read_float_safe(self.ui.vac_lineEdit_dL_G)
    dB = _read_float_safe(self.ui.vac_lineEdit_dL_B)

    dX = np.array([dR, dG, dB], dtype=np.float32)  # (3,)

    # 2) 사용할 gray index 가져오기
    # 예시: spin box에서 현재 gray 선택
    try:
        g = int(self.ui.vac_spin_gray.value())
    except Exception:
        # 혹시 그런 위젯이 없으면, 기본으로 128 같은 값 사용해도 됨
        g = 128

    if not hasattr(self, "_J_dense"):
        logging.error("[VAC] _J_dense(자코비안)가 없습니다.")
        return

    if g < 0 or g >= len(self._J_dense):
        logging.error(f"[VAC] invalid gray index g={g}, J_dense length={len(self._J_dense)}")
        return

    # 3) J_g 가져와서 ΔY = J_g · ΔX 계산
    Jg = np.asarray(self._J_dense[g], dtype=np.float32)

    # 혹시 shape이 (3,3)이 아니면 reshape
    Jg = Jg.reshape(3, 3)

    # ΔY_pred = [ΔCx, ΔCy, ΔGamma]
    dY = Jg @ dX
    dCx, dCy, dGamma = map(float, dY)

    logging.info(
        f"[VAC] computeY @ gray {g}: "
        f"ΔX=[ΔR={dR:.3f}, ΔG={dG:.3f}, ΔB={dB:.3f}] → "
        f"ΔY_pred=[ΔCx={dCx:+.6f}, ΔCy={dCy:+.6f}, ΔGamma={dGamma:+.6f}]"
    )

    # 4) UI 업데이트 (보기 좋게 포맷팅)
    self.ui.vac_lineEdit_dCx.setText(f"{dCx:+.6f}")
    self.ui.vac_lineEdit_dCy.setText(f"{dCy:+.6f}")
    self.ui.vac_lineEdit_dGamma.setText(f"{dGamma:+.6f}")