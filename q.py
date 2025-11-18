    # ---------------------------------------------------------
    # When user edits R/G/B (또는 Gray12)
    # ---------------------------------------------------------
    def on_table_changed(self, *args):
        """
        - 사용자가 테이블 값을 수정할 때 호출
        - R/G/B/Gray12 값들을 0~4095로 클램프
        - 클램프된 값으로 다시 테이블을 업데이트
        - 그리고 나서 그래프(update_plot) 다시 그림
        """
        # 재진입 방지 플래그 (setText()가 다시 dataChanged를 발생시키기 때문)
        if getattr(self, "_in_table_changed", False):
            return

        self._in_table_changed = True

        # 1~4열: LUT Index, R_High, G_High, B_High
        for r in range(256):
            for c in (1, 2, 3, 4):
                item = self.model.item(r, c)
                if item is None:
                    continue
                txt = item.text().strip()
                if txt == "":
                    continue

                try:
                    v = float(txt)
                except ValueError:
                    # 숫자 아닌 값이면 일단 건너뜀 (원하시면 0으로 강제해도 됨)
                    continue

                # 0 ~ 4095로 클램프
                v_clamped = max(0, min(4095, v))

                # 실제 값이 바뀌는 경우에만 setText (불필요한 dataChanged 방지)
                if v_clamped != v:
                    item.setText(str(int(v_clamped)))

        self._in_table_changed = False

        # 클램프가 끝난 후, 그래프 업데이트
        self.update_plot()