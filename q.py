# 라인 세팅
self.vac_optimization_gammalinearity_chart.set_series(
    "OFF_slope8",
    mids_off,
    slopes_off,
    marker='o',
    linestyle='-',
    label='OFF slope(8)'
)
off_ln = self.vac_optimization_gammalinearity_chart.lines["OFF_slope8"]
off_ln.set_color('black')
off_ln.set_markersize(3)   # 기존보다 작게 (기본이 6~8 정도일 가능성)

self.vac_optimization_gammalinearity_chart.set_series(
    "ON_slope8",
    mids_on,
    slopes_on,
    marker='o',
    linestyle='-',
    label='ON slope(8)'
)
on_ln = self.vac_optimization_gammalinearity_chart.lines["ON_slope8"]
on_ln.set_color('red')
on_ln.set_markersize(3)

# y축 autoscale with margin 1.1
all_slopes = np.concatenate([
    np.asarray(slopes_off, dtype=np.float64),
    np.asarray(slopes_on,  dtype=np.float64),
])
all_slopes = all_slopes[np.isfinite(all_slopes)]
if all_slopes.size > 0:
    ymin = np.min(all_slopes)
    ymax = np.max(all_slopes)
    center = 0.5*(ymin+ymax)
    half = 0.5*(ymax-ymin)
    if half <= 0:
        half = max(0.001, abs(center)*0.05)
    half *= 1.1  # 10% margin
    new_min = center - half
    new_max = center + half

    ax_slope = self.vac_optimization_gammalinearity_chart.ax
    cs.MatFormat_Axis(ax_slope,
                      min_val=np.float64(new_min),
                      max_val=np.float64(new_max),
                      tick_interval=None,
                      axis='y')
    ax_slope.relim(); ax_slope.autoscale_view(scalex=False, scaley=False)
    self.vac_optimization_gammalinearity_chart.canvas.draw()