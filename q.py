def _ensure_row_count(self, table, row_idx):
    # 기존 동작
    if table.rowCount() <= row_idx:
        old_rows = table.rowCount()
        table.setRowCount(row_idx + 1)

        # 새로 열린 구간에 대해서 header label 채우기
        vh = table.verticalHeader()
        for r in range(old_rows, row_idx + 1):
            vh_item = vh.model().headerData(r, Qt.Vertical)
            # headerData가 비어있을 때만 세팅 (중복세팅 방지)
            if vh_item is None or str(vh_item) == "":
                vh.setSectionResizeMode(r, QHeaderView.Fixed)  # optional: 높이 고정 유지
                table.setVerticalHeaderItem(r, QTableWidgetItem(str(r)))