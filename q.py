# ... _run_correction_iteration 내부에서 4096 LUT 만들고 나서:

# 1) TV 키로 새 LUT dict 구성 (Low는 캐시에서 그대로, High만 보정치 반영 예시)
new_lut_tvkeys = {
    "RchannelLow":  np.asarray(self._vac_dict_cache["RchannelLow"], dtype=np.float32),
    "GchannelLow":  np.asarray(self._vac_dict_cache["GchannelLow"], dtype=np.float32),
    "BchannelLow":  np.asarray(self._vac_dict_cache["BchannelLow"], dtype=np.float32),
    "RchannelHigh": self._up256_to_4096(high_256_new["R_High"]),  # 보정 결과
    "GchannelHigh": self._up256_to_4096(high_256_new["G_High"]),
    "BchannelHigh": self._up256_to_4096(high_256_new["B_High"]),
}

# 2) 제어필드 유지 + 6채널 교체 → TV에 쓸 JSON 생성
vac_write_json = self.build_vacparam_std_format(new_lut_tvkeys)

# 3) TV에 적용 (기존 쓰기 QThread 사용)
self._write_vac_to_tv(vac_write_json, on_finished=_after_write)

import copy
import json
import numpy as np

def build_vacparam_std_format(self, new_lut_tvkeys: dict) -> str:
    """
    self._vac_dict_cache(원본 TV JSON dict)를 베이스로, 전달된 6채널 LUT만 교체하여
    TV가 바로 읽을 수 있는 JSON 문자열을 반환.
    
    new_lut_tvkeys 예:
      {
        "RchannelLow":  [4096개 int],
        "RchannelHigh": [4096개 int],
        "GchannelLow":  ...,
        "GchannelHigh": ...,
        "BchannelLow":  ...,
        "BchannelHigh": ...
      }
    """
    if not hasattr(self, "_vac_dict_cache") or not isinstance(self._vac_dict_cache, dict):
        raise RuntimeError("VAC 캐시(_vac_dict_cache)가 비어있습니다. TV에서 VAC JSON을 먼저 읽어와 주세요.")

    out = copy.deepcopy(self._vac_dict_cache)

    channel_keys = (
        "RchannelLow", "RchannelHigh",
        "GchannelLow", "GchannelHigh",
        "BchannelLow", "BchannelHigh",
    )

    for k in channel_keys:
        if k not in new_lut_tvkeys:
            # 전달 안 된 키는 기존 값 유지
            continue
        arr = np.asarray(new_lut_tvkeys[k])
        if arr.shape != (4096,):
            raise ValueError(f"{k}: 길이는 4096이어야 합니다. (현재 {arr.shape})")
        # 정수 범위/형변환 보정 (0~4095 보장)
        arr = np.clip(np.round(arr).astype(np.int32), 0, 4095)
        out[k] = arr.tolist()

    # 제어필드는 out에 이미 보존됨
    return json.dumps(out, separators=(',', ':'))