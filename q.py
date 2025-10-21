# ── 공통 서식: CIE1976과 동일한 cs.* 사용 ──
cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
self.fig.subplots_adjust(hspace=0.0)  # 두 축 간격 완전히 제거

for i, (ax, atitle) in enumerate(((self.ax_main, 'Gamma (Main 0°)'),
                                  (self.ax_sub,  'Gamma (Sub 60°)'))):

    cs.MatFormat_FigArea(ax)

    # (1) 제목 표시: 위쪽 축만
    if i == 0:
        cs.MatFormat_ChartTitle(ax, title=atitle, color='#595959')
    else:
        cs.MatFormat_ChartTitle(ax, title=None)  # 아래쪽 제목 제거

    # (2) x축 제목 및 눈금: 아래쪽 축만
    if i == 1:
        cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x')
        ax.tick_params(axis='x', which='both', labelbottom=True)  # 눈금 표시
    else:
        cs.MatFormat_AxisTitle(ax, axis_title='', axis='x')  # 숨김
        ax.tick_params(axis='x', which='both', labelbottom=False)  # 눈금 숨김

    # (3) y축 설정
    cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y')
    cs.MatFormat_Axis(ax, min_val=0, max_val=255, tick_interval=x_tick, axis='x')
    cs.MatFormat_Axis(ax, min_val=0, max_val=1, tick_interval=0.25, axis='y')
    cs.MatFormat_Gridline(ax, linestyle='--')