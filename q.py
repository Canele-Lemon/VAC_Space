def build_vac_write_json(self, lut_std: dict) -> str:
    """
    표준 dict(R_Low/R_High/...) + controls → TV에 쓰는 원래 JSON 문자열로 변환.
    - controls는 lut_std["_controls"]에서 가져옴(ON에서 read → normalize 단계에서 저장해둔 것)
    """
    import json, numpy as np, copy

    controls = lut_std.get("_controls")
    if controls is None:
        raise RuntimeError("[VAC] controls 정보가 없어 write JSON을 만들 수 없습니다. (ON 단계 read/normalize 필요)")

    def _to_pylist4096(a):
        arr = np.asarray(a, dtype=np.float32).reshape(-1)
        if arr.size != 4096:
            x_src = np.linspace(0, 1, arr.size, dtype=np.float32)
            x_dst = np.linspace(0, 1, 4096, dtype=np.float32)
            arr = np.interp(x_dst, x_src, arr.astype(np.float64)).astype(np.float32)
        arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=4095.0, neginf=0.0), 0, 4095)
        return [int(round(v)) for v in arr.tolist()]

    # 원래 키로 매핑
    payload = {
        # --- 제어 필드 유지 ---
        "DRV_valc_major_ctrl":     controls.get("DRV_valc_major_ctrl"),
        "DRV_valc_pattern_ctrl_0": controls.get("DRV_valc_pattern_ctrl_0"),
        "DRV_valc_pattern_ctrl_1": controls.get("DRV_valc_pattern_ctrl_1"),
        "DRV_valc_sat_ctrl":       controls.get("DRV_valc_sat_ctrl"),
        "DRV_valc_hpf_ctrl_0":     controls.get("DRV_valc_hpf_ctrl_0"),
        "DRV_valc_hpf_ctrl_1":     controls.get("DRV_valc_hpf_ctrl_1"),
        # --- LUT 채널 ---
        "RchannelLow":  _to_pylist4096(lut_std["R_Low"]),
        "RchannelHigh": _to_pylist4096(lut_std["R_High"]),
        "GchannelLow":  _to_pylist4096(lut_std["G_Low"]),
        "GchannelHigh": _to_pylist4096(lut_std["G_High"]),
        "BchannelLow":  _to_pylist4096(lut_std["B_Low"]),
        "BchannelHigh": _to_pylist4096(lut_std["B_High"]),
    }
    return json.dumps(payload, separators=(',', ':'))