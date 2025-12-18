    def _set_item(self, table, row, col, value):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))

        table.scrollToItem(item, QAbstractItemView.PositionAtTop)
        
    def _set_item_with_spec(self, table, row, col, value, *, is_spec_ok: bool):
        self._ensure_row_count(table, row)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText("" if value is None else str(value))
        if is_spec_ok:
            item.setBackground(QColor(0, 0, 255))
        else:
            item.setBackground(QColor(255, 0, 0))

        table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
