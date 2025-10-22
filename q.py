# GammaChart __init__ 안의 포맷 부분만 교체
from matplotlib.ticker import MaxNLocator

# ... 기존 코드 ...
cs.MatFormat_ChartArea(self.fig, left=left, right=right, top=top, bottom=bottom)
for i, (ax, atitle) in enumerate(((self.ax_main, 'Gamma'), (self.ax_sub, 'Gamma'))):
    cs.MatFormat_FigArea(ax)
    # (1) 제목: 위쪽만
    cs.MatFormat_ChartTitle(ax, title=(atitle if i==0 else None), color='#595959')

    # (2) x축: 아래쪽만 라벨/눈금
    if i == 1:
        cs.MatFormat_AxisTitle(ax, axis_title='Gray Level', axis='x')
        ax.tick_params(axis='x', which='both', labelbottom=True)
    else:
        cs.MatFormat_AxisTitle(ax, axis_title='', axis='x')
        ax.tick_params(axis='x', which='both', labelbottom=False)

    # (3) x축 범위/틱만 고정, y축은 오토스케일
    cs.MatFormat_Axis(ax, min_val=0, max_val=255, tick_interval=x_tick, axis='x')
    # 🔽 y축은 범위 미고정: 눈금만 5개 내로 자동 배치
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    cs.MatFormat_AxisTitle(ax, axis_title='Luminance (nit)', axis='y')
    cs.MatFormat_Gridline(ax, linestyle='--')

# ── 시리즈 생성: linewidth=0.8 로 통일 ──
lw = 0.8
self._lines = {}; self._data = {}
for role, ax in (('main', self.ax_main), ('sub', self.ax_sub)):
    for pat, col in self._PAT_COLORS.items():
        k_off = ('OFF', role, pat)
        ln_off, = ax.plot([], [], linestyle='--', color=col, linewidth=lw, label=f'OFF {role} {pat}')
        self._lines[k_off] = ln_off; self._data[k_off] = {'x':[], 'y':[]}
        k_on = ('ON', role, pat)
        ln_on,  = ax.plot([], [], linestyle='-',  color=col, linewidth=lw, label=f'ON {role} {pat}')
        self._lines[k_on] = ln_on;  self._data[k_on]  = {'x':[], 'y':[]}
        
class CIE1976Chart:
    def __init__(...):
        ...
        # 시리즈 4개를 '상태×역할'로 생성
        ms = 3.5  # 작은 점
        self.lines = {
            ('OFF','main'): self.ax.plot([], [], 'o', markersize=ms,
                                         markerfacecolor='none', markeredgecolor='red',   linewidth=0)[0],
            ('OFF','sub'):  self.ax.plot([], [], 's', markersize=ms,
                                         markerfacecolor='none', markeredgecolor='red',   linewidth=0)[0],
            ('ON','main'):  self.ax.plot([], [], 'o', markersize=ms,
                                         markerfacecolor='none', markeredgecolor='green', linewidth=0)[0],
            ('ON','sub'):   self.ax.plot([], [], 's', markersize=ms,
                                         markerfacecolor='none', markeredgecolor='green', linewidth=0)[0],
        }
        self.data = {k:{'u':[],'v':[]} for k in self.lines.keys()}
        self._update_legend()

    def reset_on(self):
        for k in (('ON','main'), ('ON','sub')):
            self.data[k]['u'].clear(); self.data[k]['v'].clear()
            self.lines[k].set_data([], [])
        self._update_legend(); self.canvas.draw_idle()

    # 호출부에서 state='OFF'|'ON', role='main'|'sub' 로 부릅니다.
    def add_point(self, *, state:str, role:str, u_p:float, v_p:float):
        key = (state, role)
        if key not in self.lines: return
        self.data[key]['u'].append(float(u_p))
        self.data[key]['v'].append(float(v_p))
        self.lines[key].set_data(self.data[key]['u'], self.data[key]['v'])
        # 범례 라벨(각 시리즈 0°/60° 표시는 역할로 구분)
        pretty = f"{state} {role}"
        self.lines[key].set_label(pretty)
        self._update_legend(); self.canvas.draw_idle()

    def _update_legend(self):
        # 기준선 + 데이터(있는 것만)
        handles, labels = [], []
        for ln in self.ax.lines:
            lb = ln.get_label()
            if lb in ("BT.709", "DCI"):
                handles.append(ln); labels.append(lb)
        for k in (('OFF','main'),('OFF','sub'),('ON','main'),('ON','sub')):
            ln = self.lines.get(k)
            if ln is not None and ln.get_xdata() and ln.get_ydata():
                handles.append(ln); labels.append(ln.get_label())
        if handles:
            self.ax.legend(handles, labels, fontsize=8, loc='lower right')  # 🔸 폰트 8
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()
              
              
from PySide2.QtWidgets import QAbstractItemView, QTableWidgetItem

def _set_item(self, table, row, col, value):
    self._ensure_row_count(table, row)
    item = table.item(row, col)
    if item is None:
        item = QTableWidgetItem()
        table.setItem(row, col, item)
    item.setText("" if value is None else str(value))

    # 🔸 업데이트된 셀로 스크롤(보이도록)
    table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
    
from PySide2.QtGui import QColor

def _set_item_with_spec(self, table, row, col, value, *, is_spec_ok: bool):
    self._ensure_row_count(table, row)
    item = table.item(row, col)
    if item is None:
        item = QTableWidgetItem()
        table.setItem(row, col, item)
    item.setText("" if value is None else str(value))
    # 🔸 스펙 OUT만 빨간 배경
    if is_spec_ok:
        item.setBackground(QColor(255, 255, 255))  # 기본(흰색)로 돌림
    else:
        item.setBackground(QColor(255, 200, 200))  # 연한 빨강

    table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
    

              

