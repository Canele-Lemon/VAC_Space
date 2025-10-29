def _scroll_show_column_window(self, table, lead_col, window=3):
    """
    table    : QTableWidget
    lead_col : 우리가 "마지막으로 갱신한 열 index"
               예: OFF일 땐 2, ON일 땐 8
    window   : 한 화면에 보여주고 싶은 열 개수 (지금 요구는 3열씩)

    동작:
      - start_col = max(lead_col - (window-1), 0)
        예: lead_col=2, window=3 → start_col=0
            lead_col=8, window=3 → start_col=6
      - 해당 start_col의 왼쪽 edge pixel을 구해서
        horizontalScrollBar().setValue()로 그 위치로 스크롤
    """
    if table is None:
        return

    header = table.horizontalHeader()
    scrollbar = table.horizontalScrollBar()

    # 우리가 보이게 하고 싶은 첫 열
    start_col = lead_col - (window - 1)
    if start_col < 0:
        start_col = 0

    # start_col의 왼쪽 픽셀 offset을 구한다.
    # header.sectionPosition(c)는 그 열의 왼쪽 x좌표(스크롤 0 기준) px를 줌.
    try:
        target_x = header.sectionPosition(start_col)
    except Exception:
        # 만약 header가 아직 레이아웃 안 된 타이밍이면 그냥 리턴(나중에 다시 불릴 수도 있음)
        return

    # 스크롤바를 그 위치로 맞춰준다
    scrollbar.setValue(target_x)