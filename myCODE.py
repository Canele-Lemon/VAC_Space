def build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict = None) -> str:
    """
    base_vac_dict : TV에서 읽은 원본 JSON(dict; 키 순서 유지 권장)
    new_lut_tvkeys: 교체할 LUT만 전달 시 병합 (TV 원 키명 그대로)
                    {"RchannelLow":[...4096], "RchannelHigh":[...], ...}
    return: TV에 바로 쓸 수 있는 탭 포맷 문자열
    """
    from collections import OrderedDict
    import numpy as np, json

    if not isinstance(base_vac_dict, (dict, OrderedDict)):
        raise ValueError("base_vac_dict must be dict/OrderedDict")

    od = OrderedDict(base_vac_dict)

    # 새 LUT 반영(형태/범위 보정)
    if new_lut_tvkeys:
        for k, v in new_lut_tvkeys.items():
            if k in od:
                arr = np.asarray(v)
                if arr.shape != (4096,):
                    raise ValueError(f"{k}: 4096 길이 필요 (현재 {arr.shape})")
                od[k] = np.clip(arr.astype(int), 0, 4095).tolist()

    # -------------------------------
    # 포맷터
    # -------------------------------
    def _fmt_inline_list(lst):
        # [\t1,\t2,\t...\t]
        return "[\t" + ",\t".join(str(int(x)) for x in lst) + "\t]"

    def _fmt_list_of_lists(lst2d):
        """
        2D 리스트(예: DRV_valc_pattern_ctrl_1) 전용.
        마지막 닫힘은 ‘]\t\t]’ (쉼표 없음). 쉼표는 바깥 루프에서 1번만 붙임.
        """
        if not lst2d:
            return "[\t]"
        if not isinstance(lst2d[0], (list, tuple)):
            return _fmt_inline_list(lst2d)

        lines = []
        # 첫 행
        lines.append("[\t[\t" + ",\t".join(str(int(x)) for x in lst2d[0]) + "\t],")
        # 중간 행들
        for row in lst2d[1:-1]:
            lines.append("\t\t\t[\t" + ",\t".join(str(int(x)) for x in row) + "\t],")
        # 마지막 행(쉼표 없음) + 닫힘 괄호 정렬: “]\t\t]”
        last = "\t\t\t[\t" + ",\t".join(str(int(x)) for x in lst2d[-1]) + "\t]\t\t]"
        lines.append(last)
        return "\n".join(lines)

    def _fmt_flat_4096(lst4096):
        """
        4096 길이 LUT을 256x16으로 줄바꿈.
        마지막 줄은 ‘\t\t]’로 끝(쉼표 없음). 쉼표는 바깥에서 1번만.
        """
        a = np.asarray(lst4096, dtype=int)
        if a.size != 4096:
            raise ValueError(f"LUT 길이는 4096이어야 합니다. (현재 {a.size})")
        rows = a.reshape(256, 16)

        out = []
        # 첫 줄
        out.append("[\t" + ",\t".join(str(x) for x in rows[0]) + ",")
        # 중간 줄
        for r in rows[1:-1]:
            out.append("\t\t\t" + ",\t".join(str(x) for x in r) + ",")
        # 마지막 줄 (쉼표 X) + 닫힘
        out.append("\t\t\t" + ",\t".join(str(x) for x in rows[-1]) + "\n\t\t]")
        return "\n".join(out)

    lut_keys_4096 = {
        "RchannelLow","RchannelHigh",
        "GchannelLow","GchannelHigh",
        "BchannelLow","BchannelHigh",
    }

    # -------------------------------
    # 본문 생성
    # -------------------------------
    keys = list(od.keys())
    lines = ["{"]

    for i, k in enumerate(keys):
        v = od[k]
        is_last_key = (i == len(keys) - 1)
        trailing = "" if is_last_key else ","

        if isinstance(v, list):
            # 4096 LUT
            if k in lut_keys_4096 and len(v) == 4096 and not (v and isinstance(v[0], (list, tuple))):
                body = _fmt_flat_4096(v)                       # 끝에 쉼표 없음
                lines.append(f"\"{k}\"\t:\t{body}{trailing}")  # 쉼표는 여기서 1번만
            else:
                # 일반 1D / 2D 리스트
                if v and isinstance(v[0], (list, tuple)):
                    body = _fmt_list_of_lists(v)               # 끝에 쉼표 없음
                    lines.append(f"\"{k}\"\t:\t{body}{trailing}")
                else:
                    body = _fmt_inline_list(v)                 # 끝에 쉼표 없음
                    lines.append(f"\"{k}\"\t:\t{body}{trailing}")

        elif isinstance(v, (int, float)):
            lines.append(f"\"{k}\"\t:\t{int(v)}{trailing}")

        else:
            # 혹시 모를 기타 타입
            body = json.dumps(v, ensure_ascii=False)
            lines.append(f"\"{k}\"\t:\t{body}{trailing}")

    lines.append("}")
    return "\n".join(lines)