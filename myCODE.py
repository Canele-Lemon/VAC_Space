def build_vacparam_std_format(self, base_vac_dict: dict, new_lut_tvkeys: dict = None) -> str:
    """
    base_vac_dict : TV에서 읽은 원본 JSON(dict; 키 순서 유지 권장)
    new_lut_tvkeys: 교체할 6개 LUT 키만 전달 시 병합
                    {"RchannelLow": [...], "RchannelHigh": [...], ...}
    return: TV에 바로 쓸 수 있는 탭 포맷 문자열 (샘플 포맷과 동일)
    """
    from collections import OrderedDict
    import numpy as np, json

    if not isinstance(base_vac_dict, (dict, OrderedDict)):
        raise ValueError("base_vac_dict must be dict/OrderedDict")

    od = OrderedDict(base_vac_dict)

    # 새 LUT 반영
    if new_lut_tvkeys:
        for k, v in new_lut_tvkeys.items():
            if k in od:
                arr = np.asarray(v)
                if arr.shape != (4096,):
                    raise ValueError(f"{k}: 4096 길이 필요 (현재 {arr.shape})")
                od[k] = np.clip(arr.astype(int), 0, 4095).tolist()

    # ----------------------------------------
    # 포맷터들
    # ----------------------------------------
    def _fmt_inline_list(lst):
        return "[\t" + ",\t".join(str(int(x)) for x in lst) + "\t]"

    def _fmt_list_of_lists(lst2d):
        """2D 패턴 리스트 전용 포맷터 — 마지막 닫힘 ‘]		],’ 포함"""
        if not lst2d:
            return "[\t]"
        if not isinstance(lst2d[0], (list, tuple)):
            return _fmt_inline_list(lst2d)

        lines = []
        # 첫 행
        lines.append("[\t[\t" + ",\t".join(str(int(x)) for x in lst2d[0]) + "\t],")
        # 중간 행
        for row in lst2d[1:-1]:
            lines.append("\t\t\t[\t" + ",\t".join(str(int(x)) for x in row) + "\t],")
        # 마지막 행 — 닫힘 괄호 정렬: "]\t\t],"
        last = "\t\t\t[\t" + ",\t".join(str(int(x)) for x in lst2d[-1]) + "\t]\t\t],"
        lines.append(last)
        return "\n".join(lines)

    def _fmt_flat_4096(lst4096):
        a = np.asarray(lst4096, dtype=int)
        if a.size != 4096:
            raise ValueError(f"LUT 길이는 4096이어야 합니다. (현재 {a.size})")
        rows = a.reshape(256, 16)
        out = []
        first_line = "[\t" + ",\t".join(str(x) for x in rows[0]) + ","
        out.append(first_line)
        for r in rows[1:-1]:
            out.append("\t\t\t" + ",\t".join(str(x) for x in r) + ",")
        out.append("\t\t\t" + ",\t".join(str(x) for x in rows[-1]) + "\n\t\t]")
        return "\n".join(out)

    lut_keys_4096 = {
        "RchannelLow","RchannelHigh",
        "GchannelLow","GchannelHigh",
        "BchannelLow","BchannelHigh",
    }

    # ----------------------------------------
    # 본문 생성
    # ----------------------------------------
    keys = list(od.keys())
    lines = ["{"]

    for i, k in enumerate(keys):
        v = od[k]
        comma = "," if i < len(keys) - 1 else ""

        if isinstance(v, list):
            # (A) 4096 LUT 포맷
            if k in lut_keys_4096 and len(v) == 4096 and not (isinstance(v[0], (list, tuple))):
                body = _fmt_flat_4096(v)
                first_newline = body.find("\n")
                if first_newline == -1:
                    lines.append(f"\"{k}\"\t:\t{body}{comma}")
                else:
                    head = body[:first_newline]
                    tail = body[first_newline+1:]
                    lines.append(f"\"{k}\"\t:\t{head}")
                    if tail:
                        lines.append(tail + comma)
                    else:
                        lines[-1] = lines[-1] + comma
            else:
                # (B) 일반 리스트 / 2D 패턴 리스트
                if v and isinstance(v[0], (list, tuple)):
                    inner = _fmt_list_of_lists(v)
                    lines.append(f"\"{k}\"\t:\t{inner}{comma}")
                else:
                    body = _fmt_inline_list(v)
                    lines.append(f"\"{k}\"\t:\t{body}{comma}")

        elif isinstance(v, (int, float)):
            lines.append(f"\"{k}\"\t:\t{int(v)}{comma}")

        else:
            body = json.dumps(v, ensure_ascii=False)
            lines.append(f"\"{k}\"\t:\t{body}{comma}")

    lines.append("}")
    return "\n".join(lines)