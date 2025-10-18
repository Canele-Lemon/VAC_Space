def _dev_zero_lut_from_file(self, path: str):
    """
    테스트용: 파일에서 VAC JSON 읽어서 LUT 6개를 모두 0으로 바꾼 후
    build_vacparam_std_format으로 포맷 변환 후 임시파일로 저장
    """
    import json, tempfile, webbrowser

    with open(path, "r", encoding="utf-8") as f:
        vac_dict = json.load(f)

    # 6개 LUT 키만 0으로 채운 dict 생성
    zeros = [0]*4096
    zero_luts = {
        "RchannelLow": zeros, "RchannelHigh": zeros,
        "GchannelLow": zeros, "GchannelHigh": zeros,
        "BchannelLow": zeros, "BchannelHigh": zeros,
    }

    # 포맷 변환
    vac_text = self.build_vacparam_std_format(base_vac_dict=vac_dict, new_lut_tvkeys=zero_luts)

    # 임시파일로 열기
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tf:
        tf.write(vac_text)
        tmp_path = tf.name
    webbrowser.open(tmp_path)