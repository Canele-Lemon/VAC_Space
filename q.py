# 기존
self.vac_optimization_cie1976_chart.add_point(
    state=state,
    role=role,
    u_p=float(u_p),
    v_p=float(v_p)
)

# 변경 👉 CIE1976Chart.update 시그니처에 맞게
data_label = 'data_1' if state == 'OFF' else 'data_2'
view_angle = 0 if role == 'main' else 60
self.vac_optimization_cie1976_chart.update(
    u_p=float(u_p),
    v_p=float(v_p),
    data_label=data_label,
    view_angle=view_angle,
    vac_status=s['profile'].legend_text  # 예: "VAC OFF (Ref.)" / "VAC ON"
)

# 기존
if ln.axes is ax and ln.get_xdata() and ln.get_ydata():
# 변경
xd, yd = ln.get_xdata(), ln.get_ydata()
if ln.axes is ax and len(xd) > 0 and len(yd) > 0:
    
    
# 기존
if r==role and ln.get_xdata() and ln.get_ydata():
# 변경
xd, yd = ln.get_xdata(), ln.get_ydata()
if r == role and len(xd) > 0 and len(yd) > 0:
    
# 기존
if ln is not None and ln.get_xdata() and ln.get_ydata():
# 변경
xd, yd = ln.get_xdata(), ln.get_ydata()
if ln is not None and len(xd) > 0 and len(yd) > 0: