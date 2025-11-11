def parse_dlut_from_vac_version(vac_version: str) -> float | None:
    """
    vac_version 문자열에서 ΔLUT 숫자만 추출.

    규칙:
      1) 먼저 [+/-]가 붙은 숫자들을 찾는다. 예: 'G+50', 'R-100', 'R+50_G+50'
         → ['+50', '-100', '+50'] 이런 식
         → 첫 번째 것만 사용.
      2) 그런 게 하나도 없으면,
         - 그냥 숫자들만 있는 경우 (예: 'G50') 에서 마지막 숫자를 사용.
         - 그래도 없으면 None.
    """
    # 1) + 또는 - 가 *반드시* 붙어 있는 숫자만 우선 검색
    signed_nums = re.findall(r'[+-]\d+', vac_version)
    if signed_nums:
        return float(signed_nums[0])

    # 2) fallback: 부호 없는 숫자 (예: 'G50')
    plain_nums = re.findall(r'\d+', vac_version)
    if plain_nums:
        # 맨 마지막 숫자를 쓰는 쪽이 보통 LUT step일 가능성이 큼
        return float(plain_nums[-1])

    return None