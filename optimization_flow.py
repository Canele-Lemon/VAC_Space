def _confirm_vac_optimization_target(self):
    model_year  = self.vac_cmb_ModelYear.currentText().strip()
    model_name  = self.vac_cmb_ModelName.currentText().strip()
    panel_maker = self.vac_cmb_PanelMaker.currentText().strip()
    frame_rate  = self.vac_cmb_FrameRate.currentText().strip()

    msg = (
        f"Is the model you want to optimize correct?\n\n"
        f"Model Year : {model_year}\n"
        f"Model Name : {model_name}\n"
        f"Panel Maker : {panel_maker}\n"
        f"Frame Rate : {frame_rate} Hz"
    )

    reply = QMessageBox.question(
        self,
        "Confirm VAC Optimization Target",
        msg,
        QMessageBox.Ok | QMessageBox.Cancel,
        QMessageBox.Cancel
    )

    return reply == QMessageBox.Ok
    
    
def _get_jacobian_filename(self):
    panel_maker = self.vac_cmb_PanelMaker.currentText().strip()
    frame_rate = self.vac_cmb_FrameRate.currentText().strip().replace("Hz", "").strip()

    jacobian_map = {
        ("CSOTCSPI", "60"):  "jacobian_bundle_CSOTCSPI_60Hz_base_lam0.001_20260601_143741.npy",
        ("HKCH2", "60"):     "jacobian_bundle_HKCH2_60Hz_base_lam0.001_20260601_143958.npy",
        ("HKCH2", "120"):    "jacobian_bundle_HKCH2_120Hz_base_lam0.001_20260601_144220.npy",
        ("INX", "60"):       "jacobian_bundle_INX_60Hz_base_lam0.001_20260601_144442.npy",
        ("INX", "120"):      "jacobian_bundle_INX_120Hz_base_lam0.001_20260601_144704.npy",
    }

    key = (panel_maker, frame_rate)

    if key not in jacobian_map:
        raise FileNotFoundError(
            f"No Jacobian file matched for PanelMaker={panel_maker}, FrameRate={frame_rate}Hz"
        )

    return jacobian_map[key]
    
jac_filename = self._get_jacobian_filename()
jac_path = cf.get_normalized_path(__file__, '.', 'models', jac_filename)