def _focus_cell(self, table: QtWidgets.QTableWidget, row: int, col: int, center: bool=True):
    """해당 셀을 선택하고 스크롤로 가시화."""
    table.setCurrentCell(row, col)  # 선택 & 포커스
    item = table.item(row, col)
    if item is None:
        return
    hint = (QtWidgets.QAbstractItemView.PositionAtCenter
            if center else QtWidgets.QAbstractItemView.PositionAtBottom)
    table.scrollToItem(item, hint)

def _flash_cell(self, table: QtWidgets.QTableWidget, row: int, col: int,
                ms: int=350, color: QtGui.QColor=QtGui.QColor(255, 236, 179)):
    """셀 배경을 잠깐 하이라이트(시각적 피드백)."""
    item = table.item(row, col)
    if item is None:
        return
    old_brush = item.background()
    item.setBackground(QtGui.QBrush(color))
    QtCore.QTimer.singleShot(ms, lambda: item.setBackground(old_brush))