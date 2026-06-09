def _resolve_artifact_key(self):
    raw_panel_maker = self.ui.vac_cmb_PanelMaker.currentText().strip()
    raw_frame_rate = self.ui.vac_cmb_FrameRate.currentText().strip().replace("Hz", "").strip()

    panel_upper = raw_panel_maker.upper()

    if "CSOT" in panel_upper:
        artifact_panel = "CSOTCSPI"
        artifact_hz = raw_frame_rate

    elif "HKC" in panel_upper:
        artifact_panel = "HKCH2"
        artifact_hz = raw_frame_rate

    elif "BOE" in panel_upper:
        if raw_frame_rate == "120":
            artifact_panel = "INX"
            artifact_hz = "60"
        elif raw_frame_rate == "60":
            artifact_panel = "HKCH2"
            artifact_hz = "60"
        else:
            raise ValueError(
                f"Unsupported BOE FrameRate: {raw_frame_rate}Hz"
            )

    elif "INX" in panel_upper:
        artifact_panel = "INX"
        artifact_hz = raw_frame_rate

    else:
        raise ValueError(
            f"Unsupported PanelMaker for artifact loading: {raw_panel_maker}"
        )

    logging.info(
        f"[ArtifactResolve] "
        f"{raw_panel_maker} {raw_frame_rate}Hz "
        f"-> {artifact_panel} {artifact_hz}Hz"
    )

    return artifact_panel, artifact_hz
    
def _get_jacobian_filename(self):

    artifact_panel, artifact_hz = self._resolve_artifact_key()

    models_dir = cf.get_normalized_path(__file__, '.', 'models')

    pattern = f"jacobian_bundle_{artifact_panel}_{artifact_hz}Hz_*.npy"

    search_path = os.path.join(models_dir, pattern)

    matched_files = glob.glob(search_path)

    if not matched_files:
        raise FileNotFoundError(
            f"No Jacobian file matched for {artifact_panel} {artifact_hz}Hz"
        )

    matched_files.sort()

    return os.path.basename(matched_files[-1])
    
    
