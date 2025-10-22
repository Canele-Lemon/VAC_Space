def _set_item(self, table: QtWidgets.QTableWidget, row: int, col: int, value):
    # 정렬 잠시 OFF (행 재배치 방지)
    sorting_on = table.isSortingEnabled()
    if sorting_on:
        table.setSortingEnabled(False)

    # 행 확보
    self._ensure_row_count(table, row)

    # 기존 아이템이 있으면 재사용(배경/폰트 유지), 없으면 생성
    item = table.item(row, col)
    if item is None:
        item = QtWidgets.QTableWidgetItem()
        table.setItem(row, col, item)
    item.setText("" if value is None else str(value))

    # 포커스 + 스크롤 + 하이라이트
    self._focus_cell(table, row, col, center=True)
    self._flash_cell(table, row, col, ms=300)

    # 정렬 복구
    if sorting_on:
        table.setSortingEnabled(True)