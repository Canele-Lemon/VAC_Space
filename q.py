x = np.arange(256)

# 1) 먼저 데이터 넣기 (색/스타일 우리가 직접 세팅)
self.vac_optimization_chromaticity_chart.set_series(
    "OFF_Cx", x, cx_off,
    marker=None,
    linestyle='--',
    label='OFF Cx'
)
self.vac_optimization_chromaticity_chart.lines["OFF_Cx"].set_color('orange')

self.vac_optimization_chromaticity_chart.set_series(
    "ON_Cx", x, cx_on,
    marker=None,
    linestyle='-',
    label='ON Cx'
)
self.vac_optimization_chromaticity_chart.lines["ON_Cx"].set_color('orange')

self.vac_optimization_chromaticity_chart.set_series(
    "OFF_Cy", x, cy_off,
    marker=None,
    linestyle='--',
    label='OFF Cy'
)
self.vac_optimization_chromaticity_chart.lines["OFF_Cy"].set_color('green')

self.vac_optimization_chromaticity_chart.set_series(
    "ON_Cy", x, cy_on,
    marker=None,
    linestyle='-',
    label='ON Cy'
)
self.vac_optimization_chromaticity_chart.lines["ON_Cy"].set_color('red')

# y축 autoscale with margin 1.1
all_y = np.concatenate([
    np.asarray(cx_off, dtype=np.float64),
    np.asarray(cx_on,  dtype=np.float64),
    np.asarray(cy_off, dtype=np.float64),
    np.asarray(cy_on,  dtype=np.float64),
])
all_y = all_y[np.isfinite(all_y)]
if all_y.size > 0:
    ymin = np.min(all_y)
    ymax = np.max(all_y)
    center = 0.5*(ymin+ymax)
    half = 0.5*(ymax-ymin)
    # half==0일 수도 있으니 최소폭을 조금 만들어주자
    if half <= 0:
        half = max(0.001, abs(center)*0.05)
    half *= 1.1  # 10% margin
    new_min = center - half
    new_max = center + half

    ax_chr = self.vac_optimization_chromaticity_chart.ax
    cs.MatFormat_Axis(ax_chr, min_val=np.float64(new_min),
                                max_val=np.float64(new_max),
                                tick_interval=None,
                                axis='y')
    ax_chr.relim(); ax_chr.autoscale_view(scalex=False, scaley=False)
    self.vac_optimization_chromaticity_chart.canvas.draw()