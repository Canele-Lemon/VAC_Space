def _smooth_and_monotone(self, arr, win=9):
    """
    고주파(지글지글) 제거:
    1) 이동평균으로 부드럽게
    2) 단조 증가 강제 (enforce_monotone보다 먼저 완만화해서 계단 줄이기)
    arr: np.array(float32, len=256, 0~4095 스케일)
    """
    arr = np.asarray(arr, dtype=np.float32)
    half = win // 2
    tmp = np.empty_like(arr)
    n = len(arr)
    for i in range(n):
        i0 = max(0, i-half)
        i1 = min(n, i+half+1)
        tmp[i] = np.mean(arr[i0:i1])
    # 단조 정리 (non-decreasing)
    for i in range(1, n):
        if tmp[i] < tmp[i-1]:
            tmp[i] = tmp[i-1]
    return tmp


def _fix_low_high_order(self, low_arr, high_arr):
    """
    각 gray마다 Low > High이면 둘 다 중간값(mid)로 맞춰서 역전 없애기.
    반환 (low_fixed, high_fixed)
    """
    low  = np.asarray(low_arr , dtype=np.float32).copy()
    high = np.asarray(high_arr, dtype=np.float32).copy()
    for g in range(len(low)):
        if low[g] > high[g]:
            mid = 0.5 * (low[g] + high[g])
            low[g]  = mid
            high[g] = mid
    return low, high


def _nudge_midpoint(self, low_arr, high_arr, max_err=3.0, strength=0.5):
    """
    (Low+High)/2 평균 밝기가 gray(이상적으로 y=x VAC OFF)에서 너무 벗어난 곳만
    살짝 당겨서 감마 튐 억제.
    - max_err: 허용 오차(12bit count). 그 이상만 수정
    - strength: 보정 강도 (0.5면 에러의 절반만 교정)
    반환 (low_adj, high_adj)
    """
    low  = np.asarray(low_arr , dtype=np.float32).copy()
    high = np.asarray(high_arr, dtype=np.float32).copy()

    gray_12 = (np.arange(256, dtype=np.float32) * 4095.0) / 255.0
    avg     = 0.5 * (low + high)
    err     = avg - gray_12  # 양수면 평균이 너무 밝음

    mask = np.abs(err) > max_err
    adj  = err * strength   # 양수면 아래로 당김

    high[mask] -= adj[mask]
    low [mask] -= adj[mask]

    return low, high


def _finalize_channel_pair_safely(self, low_arr, high_arr):
    """
    마지막 안전화 단계:
    1) 다시 Low>High 방지
    2) 단조 증가 강제 (_enforce_monotone)
    3) 0/255 엔드포인트 강제: 0→0, 255→4095
    4) 0~4095 clip
    """
    low  = np.asarray(low_arr , dtype=np.float32).copy()
    high = np.asarray(high_arr, dtype=np.float32).copy()

    # (1) 다시 Low>High 방지
    for g in range(len(low)):
        if low[g] > high[g]:
            mid = 0.5 * (low[g] + high[g])
            low[g]  = mid
            high[g] = mid

    # (2) 단조 증가
    low  = self._enforce_monotone(low)
    high = self._enforce_monotone(high)

    # (3) 엔드포인트 고정
    low[0]  = 0.0
    high[0] = 0.0
    low[-1]  = 4095.0
    high[-1] = 4095.0

    # (4) clip
    low  = np.clip(low ,  0.0, 4095.0)
    high = np.clip(high, 0.0, 4095.0)

    return low.astype(np.float32), high.astype(np.float32)
    
    