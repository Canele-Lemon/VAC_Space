# 6) 256→4096 업샘플 (Low는 그대로, High만 갱신)
new_lut_4096 = {
    "RchannelLow":  np.asarray(vac_dict["RchannelLow"], dtype=np.float32),
    "GchannelLow":  np.asarray(vac_dict["GchannelLow"], dtype=np.float32),
    "BchannelLow":  np.asarray(vac_dict["BchannelLow"], dtype=np.float32),
    "RchannelHigh": self._up256_to_4096(high_R),
    "GchannelHigh": self._up256_to_4096(high_G),
    "BchannelHigh": self._up256_to_4096(high_B),
}

# ✅ 정수 변환 (float → int) + 범위 클리핑
for k in new_lut_4096:
    new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)