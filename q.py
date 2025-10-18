# ... 보정 계산 끝나고 high_256_new 만들었다고 가정
new_lut_tvkeys = {
    "RchannelLow":  self._vac_dict_cache["RchannelLow"],
    "GchannelLow":  self._vac_dict_cache["GchannelLow"],
    "BchannelLow":  self._vac_dict_cache["BchannelLow"],
    "RchannelHigh": self._up256_to_4096(high_256_new["R_High"]),
    "GchannelHigh": self._up256_to_4096(high_256_new["G_High"]),
    "BchannelHigh": self._up256_to_4096(high_256_new["B_High"]),
}

# 캐시 원본을 인자로 넣어 JSON 생성 (캐시는 아직 건드리지 않음)
vac_write_json = self.build_vacparam_std_format(self._vac_dict_cache, new_lut_tvkeys)

def _after_write(ok, msg):
    logging.info(f"[VAC Write] {msg}")
    if not ok:
        return
    # 쓰기 성공 → 재읽기
    self._read_vac_from_tv(_after_read_back)

def _after_read_back(vac_dict_after):
    if not vac_dict_after:
        logging.error("보정 후 VAC 재읽기 실패")
        return
    # ✅ 여기서 캐시 갱신 (성공 케이스에만)
    self._vac_dict_cache = vac_dict_after
    # 차트용 변환 후 표시
    lut_dict_plot = {k.replace("channel","_"): v
                     for k, v in vac_dict_after.items() if "channel" in k}
    self._update_lut_chart_and_table(lut_dict_plot)
    # 다음 측정 세션 시작 등...

# TV에 적용
self._write_vac_to_tv(vac_write_json, on_finished=_after_write)


import copy, json, numpy as np

def build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict) -> str:
    """
    base_vac_dict: TV에서 읽은 원본 JSON(dict) - 제어필드 포함, TV 키명 그대로
    new_lut_tvkeys: 교체할 6채널만 (TV 키명 그대로)
      { "RchannelLow": [...4096], "RchannelHigh": [...],
        "GchannelLow": [...],    "GchannelHigh": [...],
        "BchannelLow": [...],    "BchannelHigh": [...] }
    """
    if not isinstance(base_vac_dict, dict):
        raise ValueError("base_vac_dict must be dict (TV 원본 JSON)")

    out = copy.deepcopy(base_vac_dict)

    for k in (
        "RchannelLow","RchannelHigh",
        "GchannelLow","GchannelHigh",
        "BchannelLow","BchannelHigh",
    ):
        if k in new_lut_tvkeys:
            arr = np.asarray(new_lut_tvkeys[k])
            if arr.shape != (4096,):
                raise ValueError(f"{k}: 길이는 4096이어야 합니다. (현재 {arr.shape})")
            out[k] = np.clip(np.round(arr).astype(np.int32), 0, 4095).tolist()

    return json.dumps(out, separators=(',', ':'))