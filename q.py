def prepare_X_delta_raw_with_mapping(self, ref_vac_set_pk: int):
    """
    ref_vac_set_pk: W_VAC_SET_Info 테이블에서 'ref' LUT가 들어있는 세트 PK (예: 2582)
    return:
        {
        "lut_delta_raw": { "R_Low": (256,), ... },   # raw 12bit (target - ref) @ mapped j
        "meta": { panel_maker one-hot, frame_rate, model_year },
        "mapping_j": (256,) np.int32
        }
    """
    # 1) 타겟 메타/타겟 LUT(4096)
    info = self._load_vac_set_info_row(self.PK)
    if info is None:
        raise RuntimeError(f"[X_delta_raw] No VAC_SET_Info for PK={self.PK}")

    meta_dict = info["meta"]

    # 현재 세트 PK(self.PK)의 LUT
    lut4096_target = self._load_vacdata_lut4096(self.PK)
    if lut4096_target is None:
        raise RuntimeError(f"[X_delta_raw] No VAC_Data for VAC_SET_Info.PK={self.PK}")

    # 2) ref LUT(4096) — ref 세트 PK로부터
    lut4096_ref = self._load_vacdata_lut4096(ref_vac_set_pk)
    if lut4096_ref is None:
        raise RuntimeError(f"[X_delta_raw] No REF VAC_Data for VAC_SET_Info.PK={ref_vac_set_pk}")

    # 3) i(0..255) → j(0..4095) 매핑 로드 (CSV)
    j_map = self._load_lut_index_mapping()  # (256,) int

    # 4) 매핑 지점에서 raw delta(target - ref) 계산 (정규화 없음)
    delta = {}
    for ch in ['R_Low','R_High','G_Low','G_High','B_Low','B_High']:
        tgt = np.asarray(lut4096_target[ch], dtype=np.float32)
        ref = np.asarray(lut4096_ref[ch],    dtype=np.float32)
        if tgt.shape[0] != 4096 or ref.shape[0] != 4096:
            raise ValueError(f"[X_delta_raw] channel {ch}: need 4096-length arrays.")
        delta[ch] = (tgt[j_map] - ref[j_map]).astype(np.float32)

    return {"lut_delta_raw": delta, "meta": meta_dict, "mapping_j": j_map}