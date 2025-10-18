import json
import numpy as np
from collections import OrderedDict

def build_vacparam_std_format(base_vac_dict: dict,
                              new_lut_tvkeys: dict = None,
                              zero_luts: bool = False) -> str:
    """
    base_vac_dict : TV에서 읽은 원본 JSON(dict/OrderedDict) – 제어필드 포함, TV 키명 그대로(키 순서 유지)
    new_lut_tvkeys: 교체할 LUT 6키(TV 키명 그대로)만 전달 시 병합
                    {"RchannelLow": [...4096], "RchannelHigh": [...], ..., "BchannelHigh": [...]}
    zero_luts     : True면 6개 LUT 모두 0으로 설정(테스트용)

    return: TV에 바로 쓸 수 있는 탭 스타일 JSON 문자열
            (첫 줄 '{'만, 이후 "KEY"\t:\tVALUE, 4096→256×16 줄바꿈, 마지막 쉼표 없음)
    """

    if not isinstance(base_vac_dict, (dict, OrderedDict)):
        raise ValueError("base_vac_dict must be dict/OrderedDict")

    # 키 순서 보존 (파이썬 3.7+ dict도 삽입순서를 유지하지만 안전하게)
    od = OrderedDict(base_vac_dict)

    lut_keys = ("RchannelLow","RchannelHigh","GchannelLow","GchannelHigh","BchannelLow","BchannelHigh")

    # 6개 LUT 전부 0으로(옵션)
    if zero_luts:
        zeros = [0]*4096
        for k in lut_keys:
            if k in od: od[k] = zeros

    # 일부/전체 LUT 교체(옵션)
    if new_lut_tvkeys:
        for k in lut_keys:
            if k in new_lut_tvkeys:
                arr = np.asarray(new_lut_tvkeys[k])
                if arr.shape != (4096,):
                    raise ValueError(f"{k}: 길이는 4096이어야 합니다 (현재 {arr.shape}).")
                od[k] = np.clip(arr.astype(int), 0, 4095).tolist()

    # ---------- 포맷터 ----------
    def _fmt_inline_list(lst):
        # [\t1,\t2,\t3\t]
        return "[\t" + ",\t".join(str(int(x)) for x in lst) + "\t]"

    def _fmt_rows(rows, indent="\t\t\t"):
        #   \t\t\t[\t...16...\t],  (마지막 행은 콤마 없음)
        out = []
        for i, row in enumerate(rows):
            body = ",\t".join(str(int(x)) for x in row)
            comma = "," if i < len(rows)-1 else ""
            out.append(f"{indent}[\t{body}\t]{comma}")
        return "\n".join(out)

    def _fmt_2d_8x16(mat):
        # DRV_valc_pattern_ctrl_1 전용(8×16)
        m = np.asarray(mat, dtype=int).reshape(8, 16)
        first = ",\t".join(str(int(x)) for x in m[0])
        lines = [f"[\t[\t{first}\t],"]
        if m.shape[0] > 1:
            lines.append(_fmt_rows(m[1:], indent="\t\t\t"))
        lines.append("\t\t]")
        return "\n".join(lines)

    def _fmt_lut_4096(arr):
        # 4096 → 256×16
        a = np.asarray(arr, dtype=int).reshape(256, 16)
        first = ",\t".join(str(int(x)) for x in a[0])
        lines = [f"[\t{first},"]                         # 첫 행 뒤 콤마
        if a.shape[0] > 2:
            lines.append(_fmt_rows(a[1:-1], indent="\t\t\t"))
        last = ",\t".join(str(int(x)) for x in a[-1])
        lines.append(f"\t\t\t{last}\n\t\t]")              # 마지막 행은 콤마 없음
        return "\n".join(lines)
    # ---------------------------

    # 직렬화 (키 순서 유지, 콜론 앞뒤 탭, 마지막 항목 무콤마)
    out_lines = ["{"]  # 첫 줄은 '{'만
    keys = list(od.keys())
    for i, k in enumerate(keys):
        v = od[k]

        # 값 포맷 규칙
        if isinstance(v, list):
            if v and isinstance(v[0], list):
                # 2D 리스트: 8×16이면 ctrl_1 전용 포맷
                try:
                    mat = np.asarray(v)
                    if mat.shape == (8, 16):
                        body = _fmt_2d_8x16(mat)
                    else:
                        body = "[\n" + _fmt_rows(mat, indent="\t\t\t") + "\n\t\t]"
                except Exception:
                    body = _fmt_inline_list(v)
            else:
                # 1D: 4096이면 LUT 포맷, 아니면 인라인
                body = _fmt_lut_4096(v) if len(v) == 4096 else _fmt_inline_list(v)
        elif isinstance(v, (int, float)):
            body = str(int(v))
        else:
            body = json.dumps(v, ensure_ascii=False)

        comma = "," if i < len(keys)-1 else ""
        out_lines.append(f"\"{k}\"\t:\t{body}{comma}")

    out_lines.append("}")
    return "\n".join(out_lines)