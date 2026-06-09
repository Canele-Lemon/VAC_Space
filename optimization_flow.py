def _reset_vac_optimization_ui_state(self):
    # 1) 진행 중 animation 정리
    self._stop_all_step_animations()

    # 2) step icon pending 초기화
    for s in (1, 2, 3, 4, 5):
        self._step_set_pending(s)

    # 3) 차트 초기화
    try:
        self.vac_optimization_gamma_chart.reset_all()
    except Exception:
        logging.exception("[UI Reset] gamma chart reset failed")

    try:
        self.vac_optimization_cie1976_chart.reset_all()
    except Exception:
        logging.exception("[UI Reset] CIE chart reset failed")

    try:
        self.vac_optimization_lut_chart.clear()
    except Exception:
        logging.debug("[UI Reset] LUT chart clear skipped")

    # 4) 내부 상태 초기화
    self._off_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}},
                                 'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                       'colorshift': {'main': [], 'sub': []}}

    self._on_store = {'gamma': {'main': {'white':{},'red':{},'green':{},'blue':{}},
                                'sub': {'white':{},'red':{},'green':{},'blue':{}}},
                      'colorshift': {'main': [], 'sub': []}}

    self._final_vac_data_for_download = None
    self.ui.vac_btn_JSONdownload.setEnabled(False)

    self._fine_mode = False
    self._fine_ng_list = None
    
    
def reset_all(self):
    for key, ln in self._lines.items():
        self._data[key]['x'].clear()
        self._data[key]['y'].clear()
        ln.set_data([], [])

    self._autoscale()
    self._update_legends()
    self.canvas.draw_idle()
    
def reset_all(self):
    for key, ln in self.lines.items():
        if key in self.data:
            self.data[key]['u'].clear()
            self.data[key]['v'].clear()
        ln.set_data([], [])

    self._update_legend()
    self.canvas.draw_idle()
    
def _stop_step_animation(self, step: int):
    self._ensure_step_anim_map()

    if step in self._step_anim:
        try:
            label_handle, movie_handle = self._step_anim.pop(step)
            self.stop_loading_animation(label_handle, movie_handle)
        except Exception:
            logging.exception(f"[Step UI] failed to stop animation for step={step}")
            
def _stop_all_step_animations(self):
    self._ensure_step_anim_map()

    for step in list(self._step_anim.keys()):
        self._stop_step_animation(step)
        
def _step_set_pending(self, step: int):
    self._stop_step_animation(step)

    lbl = self._step_label(step)
    if lbl is None:
        return

    self._set_icon_scaled(lbl, self.process_pending_pixmap)
    
def _step_start(self, step: int):
    self._ensure_step_anim_map()

    lbl = self._step_label(step)
    if lbl is None:
        return

    self._stop_step_animation(step)

    label_handle, movie_handle = self.start_loading_animation(lbl, 'processing.gif')
    self._step_anim[step] = (label_handle, movie_handle)
    
def _step_done(self, step: int):
    self._stop_step_animation(step)

    lbl = self._step_label(step)
    if lbl is None:
        return

    self._set_icon_scaled(lbl, self.process_complete_pixmap)
    
def _step_fail(self, step: int):
    self._stop_step_animation(step)

    lbl = self._step_label(step)
    if lbl is None:
        return

    self._set_icon_scaled(lbl, self.process_fail_pixmap)
    