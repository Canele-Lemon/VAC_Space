def prepare_Y(
    self,
    y0_patterns=('W', 'R', 'G', 'B'),
    y1_patterns=('W',),
    use_y0: bool = True,
    use_y1: bool = True,
    use_y2: bool = True,
):
    """
    최종 Y 딕셔너리 병합 반환 (선택적 계산):
    - use_y0/use_y1/use_y2 플래그로 각 Y를 계산할지 선택
    - y0_patterns, y1_patterns로 사용할 패턴 제한

    예)
      - Y0(W만), Y1/Y2 미사용:
          prepare_Y(y0_patterns=('W',), use_y0=True, use_y1=False, use_y2=False)

      - 이전과 동일하게 Y0/Y1/Y2 모두:
          prepare_Y()  # 기본값 그대로 사용
    """
    out = {}

    # Y0: 정면 dGamma/dCx/dCy
    if use_y0:
        # W/R/G/B 전체 계산
        y0_all = self.compute_Y0_struct()
        # y0_patterns에 해당하는 패턴만 잘라서 사용
        if y0_patterns is None:
            out["Y0"] = y0_all
        else:
            out["Y0"] = {p: y0_all[p] for p in y0_patterns if p in y0_all}

    # Y1: 측면 gamma linearity
    if use_y1:
        out["Y1"] = self.compute_Y1_struct(patterns=y1_patterns)

    # Y2: Macbeth Δu'v'
    if use_y2:
        out["Y2"] = self.compute_Y2_struct()

    return out