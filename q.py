Traceback (most recent call last):
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 1597, in start_VAC_optimization
    self._run_off_baseline_then_on()
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 878, in _run_off_baseline_then_on
    'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"VAC OFF (Ref.) - {p}")
  File "d:\LCM_DX\OMS_2\he_opticalmeasurement\subpages\vacspace_130\subpage_vacspace.py", line 878, in <dictcomp>
    'main': {p: self.vac_optimization_gamma_chart.add_series(axis_index=0, label=f"VAC OFF (Ref.) - {p}")
AttributeError: 'GammaChart' object has no attribute 'add_series'
