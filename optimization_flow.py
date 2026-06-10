self.ui.vac_spinBox_correctionIterations.setRange(1, 10)
self.ui.vac_spinBox_correctionIterations.setValue(1)
self.ui.vac_spinBox_correctionIterations.setSingleStep(1)

def start_vac_optimization(self):
    if not self._check_vac_optimization_validation():
        return

    if not self._confirm_vac_optimization_target():
        return

    self._max_correction_iters = int(self.ui.vac_spinBox_correctionIterations.value())
    
    max_iters = getattr(self, "_max_correction_iters", 1)

self._spec_thread.finished.connect(
    lambda ok, metrics: self.on_spec_eval_done(
        ok, metrics,
        iter_idx=0,
        max_iters=max_iters
    )
)

