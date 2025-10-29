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