mids_off, slopes_off = _block_slopes(lv_off, g_start=88, g_stop=232, step=8)
mids_on , slopes_on  = _block_slopes(lv_on , g_start=88, g_stop=232, step=8)

avg_off = float(np.nanmean(slopes_off)) if np.isfinite(slopes_off).any() else float('nan')
avg_on  = float(np.nanmean(slopes_on )) if np.isfinite(slopes_on ).any() else float('nan')

tbl_gl = self.ui.vac_table_gammaLinearity
_set_text(tbl_gl, 1, 1, f"{avg_off:.6f}")  # 2행,2열 OFF 평균 기울기
_set_text(tbl_gl, 1, 2, f"{avg_on:.6f}")   # 2행,3열 ON  평균 기울기

self.vac_optimization_gammalinearity_chart.set_series(
    "OFF_slope8",
    mids_off,          # 예: [92,100,108,...,228]
    slopes_off,
    marker='o',
    linestyle='-',
    label='OFF slope(8)'
)
self.vac_optimization_gammalinearity_chart.set_series(
    "ON_slope8",
    mids_on,
    slopes_on,
    marker='o',
    linestyle='--',
    label='ON slope(8)'
)