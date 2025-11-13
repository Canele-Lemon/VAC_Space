FULL_POINTS = 4096
EPS_HIGH_OVER_LOW = 1.0

def _enforce_with_locks_12bit(high, low, eps=1.0):
    """
    high, low: 길이 4096 배열
    잠금 인덱스: 0, 1, 4092, 4095
      - high[0] = high[1] = 0
      - high[4092] = 4092
      - high[4095] = 4095
    그 외 인덱스:
      - high[j] >= low[j] + eps
      - 전체 단조 증가(비감소)
      - 단, 잠금 인덱스는 그대로 두고,
        잠금 인덱스 주변 값이 잠금값을 넘으면 잠금값으로 끌어내린다.
    """
    high = np.asarray(high, float).copy()
    low  = np.asarray(low,  float).copy()

    # 1) 잠금 값 정의
    LOCK_VALS = {
        0:    0.0,
        1:    0.0,
        4092: 4092.0,
        4095: 4095.0,
    }
    lock_idx = np.array(sorted(LOCK_VALS.keys()), dtype=int)

    # 2) 우선 잠금 인덱스에 원하는 값 강제 세팅
    for j, v in LOCK_VALS.items():
        if 0 <= j < FULL_POINTS:
            high[j] = v

    # 3) low+eps 제약: 잠금 인덱스를 제외한 곳에만 적용
    mask = np.ones_like(high, dtype=bool)
    mask[lock_idx] = False
    high[mask] = np.maximum(high[mask], low[mask] + eps)

    # 4) 중간 구간(2 ~ 4091)에 대해 단조 증가 강제
    #    (앞에서 뒤로 누적 max)
    start_mid = 2
    end_mid   = 4091
    high[start_mid:end_mid+1] = np.maximum.accumulate(high[start_mid:end_mid+1])

    # 5) 고계조 끝쪽이 잠금값(4092, 4095)을 넘지 않도록 clamp
    #    - 중간 구간 전체가 4092보다 클 수 없도록
    high[start_mid:4092] = np.minimum(high[start_mid:4092], high[4092])

    # 6) 4092~4095 구간도 단조 증가 + 4095 상한
    #    (4092는 이미 4092, 4095는 4095로 고정되어 있음)
    high[4092:4095] = np.maximum.accumulate(high[4092:4095])
    high[4092:4096] = np.minimum(high[4092:4096], high[4095])

    # 7) 혹시 앞쪽이 음수로 가는 경우 방어 (이론상 없겠지만)
    high[:2] = np.maximum(high[:2], 0.0)

    # 8) 마지막으로 잠금 인덱스를 한 번 더 덮어써서
    #    중간 과정에서 값이 바뀌지 않았음을 보장
    for j, v in LOCK_VALS.items():
        if 0 <= j < FULL_POINTS:
            high[j] = v

    return high