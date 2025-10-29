import os
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator  # monotone-ish 1D spline

# ===== 사용자 설정 =====
ref_dir = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\3. Perturbation\Ref"
output_dir = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\3. Perturbation\CSV\임시"
os.makedirs(output_dir, exist_ok=True)

NUM_SAMPLE = 100

# 저/중/고 계조에서 얼마나 크게 움직일지 (상대 변화량 최대치)
LOW_RATE  = 0.10
MID_RATE  = 0.05
HIGH_RATE = 0.02

# gray 구간 나눔 (12bit gray 기준)
def _zone_rate(gray12):
    if gray12 <= 1023:
        return LOW_RATE
    elif gray12 <= 3071:
        return MID_RATE
    else:
        return HIGH_RATE

# per-step 최대 증가량 제한 (banding 방지)
MAX_STEP = 4  # 한 gray 증가할 때 LUT 값이 이 값보다 많이 점프하지 않도록 clamp

# coarse anchor(저주파용). 끝점(0,4095)는 반드시 포함해야 monotone/경계조건 보존 쉬움
COARSE_GRAY = np.array([
    0,
    512,
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4095
], dtype=np.int32)

# --------------------------------------------------------------------------------
# helper 1: 단조 enforce (y[i] >= y[i-1])
def enforce_monotone(arr):
    out = np.asarray(arr, dtype=np.int32).copy()
    for i in range(1, len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

# helper 2: per-step slope 제한
def enforce_max_step(arr, max_step=4):
    out = np.asarray(arr, dtype=np.int32).copy()
    for i in range(1, len(out)):
        allowed = out[i-1] + max_step
        if out[i] > allowed:
            out[i] = allowed
    return out

# helper 3: Low < High 보정
def enforce_low_lt_high(low_arr, high_arr):
    low_arr  = np.asarray(low_arr,  dtype=np.int32).copy()
    high_arr = np.asarray(high_arr, dtype=np.int32).copy()
    # 조건이 깨지면 High를 살짝 올리거나, 최소 +1이라도 띄워줌
    for i in range(len(low_arr)):
        if high_arr[i] <= low_arr[i]:
            high_arr[i] = low_arr[i] + 1
    return low_arr, high_arr

# helper 4: coarse anchor 값을 만들 때, RGB를 어느 정도 같이 움직이게 하는 생성기
def build_coarse_variant(base_vals_at_coarse, lut_type_group="low_or_high"):
    """
    base_vals_at_coarse: shape (len(COARSE_GRAY), 3) = [R,G,B] at those coarse gray idx
                         (12bit 범위 값: 0~4095)
    lut_type_group: "low" or "high" or "low_or_high" (그룹 구분용, rate는 zone에 따라 다시 계산)

    return:
        variant_vals_coarse: same shape, perturbed but still int-ish (float 중간 계산 후 round)
    """

    # RGB 동조 방향(같이 들리냐/같이 내려앉냐)
    # ex) -1이면 전체적으로 살짝 낮추는 쪽, +1이면 살짝 올리는 쪽
    group_dir = np.random.choice([-1, 1])

    # 각 채널별로 약간의 개별성 부여 (0.8~1.2 배 같은 느낌)
    channel_jitter = {
        "R": np.random.uniform(0.8, 1.2),
        "G": np.random.uniform(0.8, 1.2),
        "B": np.random.uniform(0.8, 1.2),
    }

    # coarse 지점마다 perturbation
    out = np.zeros_like(base_vals_at_coarse, dtype=np.float32)  # (Ncoarse,3)
    for i, gray12 in enumerate(COARSE_GRAY):
        # 구간별 scale (저계조는 크게, 고계조는 작게)
        rate_here = _zone_rate(gray12)

        # 각 채널에 대해 적용
        for c_idx, ch in enumerate(["R","G","B"]):
            base_v = float(base_vals_at_coarse[i, c_idx])
            # perturb 비율
            # factor = 1 + direction * jitter * random(0~rate)
            factor = 1.0 + group_dir * channel_jitter[ch] * np.random.uniform(0.0, rate_here)
            new_v = base_v * factor

            # 범위 클램프
            new_v = np.clip(new_v, 0.0, 4095.0)

            out[i, c_idx] = new_v

    # 첫/끝점 고정 강제: gray=0 -> 0, gray=4095 -> 4095 근처 유지
    #   low/high LUT들 모두 HW에서 보통 0 -> 0, 4095 -> 4095 근처라는 가정이 있다면
    #   여기서도 맞춰줍니다.
    out[0, :]     = [0.0, 0.0, 0.0]
    out[-1, :]    = [4095.0, 4095.0, 4095.0]

    return np.round(out).astype(np.int32)

# helper 5: coarse → dense(256 anchor gray들) 보간
def spline_expand_to_256(coarse_gray, coarse_rgb_vals):
    """
    coarse_gray: (Nc,)
    coarse_rgb_vals: (Nc,3)  int32-ish [R,G,B] at those coarse_gray

    return:
      dense_gray_256: (256,) 12bit gray indices subsampled every ~16
      dense_vals_256: (256,3) at those 256 grays, using PCHIP
    """
    # we'll reuse the 16-step sampled grid similar to original:
    dense_gray_256 = np.round(np.linspace(0, 4095, 256)).astype(np.int32)

    dense_vals_256 = np.zeros((256,3), dtype=np.float32)
    for c_idx in range(3):
        # monotone-ish spline per channel
        spl = PchipInterpolator(coarse_gray, coarse_rgb_vals[:, c_idx])
        dense_vals_256[:, c_idx] = spl(dense_gray_256)

    # 클램프
    dense_vals_256 = np.clip(np.round(dense_vals_256), 0, 4095).astype(np.int32)
    # 첫/끝 보정
    dense_vals_256[0,:]   = [0,0,0]
    dense_vals_256[-1,:]  = [4095,4095,4095]

    return dense_gray_256, dense_vals_256

# helper 6: dense 256 anchor → full 4096 LUT (선형 보간)
def expand_256_to_4096(dense_gray_256, dense_vals_256):
    """
    dense_gray_256: (256,) 12bit gray indices
    dense_vals_256: (256,3) R,G,B
    return:
      full_gray_4096: (4096,)
      full_vals_4096: (4096,3)
    """
    full_gray = np.arange(0, 4096, dtype=np.int32)
    full_vals = np.zeros((4096,3), dtype=np.float32)

    for c_idx in range(3):
        # piecewise-linear
        full_vals[:, c_idx] = np.interp(
            full_gray,
            dense_gray_256.astype(np.float32),
            dense_vals_256[:, c_idx].astype(np.float32)
        )

    full_vals = np.clip(np.round(full_vals), 0, 4095).astype(np.int32)
    # 강제 경계
    full_vals[0,:]    = [0,0,0]
    full_vals[-1,:]   = [4095,4095,4095]

    return full_gray, full_vals

# helper 7: 후처리 전체 파이프라인 (monotone, step limit, low<high)
def postprocess_full_lut(low_rgb_4096, high_rgb_4096):
    """
    low_rgb_4096 : (4096,3)  R,G,B
    high_rgb_4096: (4096,3)  R,G,B

    return (low_pp, high_pp) processed
    """
    low_pp  = np.zeros_like(low_rgb_4096,  dtype=np.int32)
    high_pp = np.zeros_like(high_rgb_4096, dtype=np.int32)

    for c_idx in range(3):
        # 1) monotone
        lo = enforce_monotone(low_rgb_4096[:, c_idx])
        hi = enforce_monotone(high_rgb_4096[:, c_idx])

        # 2) step 제한
        lo = enforce_max_step(lo, MAX_STEP)
        hi = enforce_max_step(hi, MAX_STEP)

        # 임시 저장
        low_pp[:,  c_idx] = lo
        high_pp[:, c_idx] = hi

    # 3) Low < High 조건 (채널별)
    for c_idx in range(3):
        lo, hi = enforce_low_lt_high(low_pp[:, c_idx], high_pp[:, c_idx])
        low_pp[:,  c_idx] = lo
        high_pp[:, c_idx] = hi

    # 4) 경계값 다시 강제
    low_pp[0,:]   = [0,0,0]
    high_pp[0,:]  = np.maximum(low_pp[0,:] + 1, [1,1,1])
    low_pp[-1,:]  = [4095,4095,4095]
    high_pp[-1,:] = np.maximum(low_pp[-1,:] + 1, [4095,4095,4095])

    high_pp = np.clip(high_pp, 0, 4095)
    return low_pp, high_pp


# ================= 메인 루프 =================
for file_name in os.listdir(ref_dir):
    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(ref_dir, file_name)

    # 구분자 자동 감지 (탭 or 콤마)
    with open(file_path, "r", encoding="utf-8") as ftmp:
        first_line = ftmp.readline()
    sep_guess = "\t" if "\t" in first_line else ","

    df = pd.read_csv(file_path, sep=sep_guess)

    # 기대 컬럼:
    # Gray Level, R_Low, R_High, G_Low, G_High, B_Low, B_High
    gray_levels = df['Gray Level'].values.astype(np.int32)

    # 원본 LUT 전체 4096개 값
    R_low_base  = df['R_Low' ].values.astype(np.int32)
    R_high_base = df['R_High'].values.astype(np.int32)
    G_low_base  = df['G_Low' ].values.astype(np.int32)
    G_high_base = df['G_High'].values.astype(np.int32)
    B_low_base  = df['B_Low' ].values.astype(np.int32)
    B_high_base = df['B_High'].values.astype(np.int32)

    # coarse 지점에서의 원본 값 뽑기
    # (COARSE_GRAY는 gray_levels와 동일 스케일(0~4095)라 가정)
    # 우리가 읽은 ref CSV가 정확히 gray=0..4095 순서라고 가정
    coarse_idx = COARSE_GRAY  # 이미 정수 인덱스
    base_low_coarse  = np.stack([
        R_low_base [coarse_idx],
        G_low_base [coarse_idx],
        B_low_base [coarse_idx]
    ], axis=1)  # shape (Nc,3) but actually (Nc,RGB) -> we'll transpose later if needed
    # 위는 (Nc,3) = [R,G,B] 맞추고 싶으니 transpose 살짝 조정
    # 현재 stack은 (3,Nc) -> axis=1 해서 (Nc,3) 됨. OK.

    base_high_coarse = np.stack([
        R_high_base[coarse_idx],
        G_high_base[coarse_idx],
        B_high_base[coarse_idx]
    ], axis=1)

    # 샘플 생성
    for sample_num in range(1, NUM_SAMPLE+1):

        # ── 1) coarse 레벨에서 RGB 동조 perturbation 주기 (low, high 따로)
        pert_low_coarse  = build_coarse_variant(base_low_coarse,  lut_type_group="low")
        pert_high_coarse = build_coarse_variant(base_high_coarse, lut_type_group="high")

        # ── 2) coarse → 256 anchor (monotone-ish spline)
        dense_gray_256_low,  dense_vals_256_low  = spline_expand_to_256(COARSE_GRAY, pert_low_coarse)
        dense_gray_256_high, dense_vals_256_high = spline_expand_to_256(COARSE_GRAY, pert_high_coarse)

        # ── 3) 256 anchor → full 4096 선형 보간
        full_gray_low,  full_vals_low  = expand_256_to_4096(dense_gray_256_low,  dense_vals_256_low)
        full_gray_high, full_vals_high = expand_256_to_4096(dense_gray_256_high, dense_vals_256_high)

        # full_vals_* shape: (4096,3) = [R,G,B]

        # ── 4) 후처리: monotone, step limit, Low<High, 경계강제
        low_pp, high_pp = postprocess_full_lut(full_vals_low, full_vals_high)
        # low_pp[:,0]=R_low, low_pp[:,1]=G_low, low_pp[:,2]=B_low
        # high_pp[:,0]=R_high, ...

        # 최종 DataFrame 생성 (4096행)
        out_df = pd.DataFrame({
            "Gray Level": full_gray_low,  # 0..4095
            "R_Low":      low_pp[:,0],
            "R_High":     high_pp[:,0],
            "G_Low":      low_pp[:,1],
            "G_High":     high_pp[:,1],
            "B_Low":      low_pp[:,2],
            "B_High":     high_pp[:,2],
        })

        # ── 5) 저장
        ref_base_name = os.path.splitext(file_name)[0]
        out_file_name = f"LUT_Perturbation_smooth_{ref_base_name}_{sample_num:03d}.csv"
        out_path = os.path.join(output_dir, out_file_name)
        out_df.to_csv(out_path, index=False)

print("[✅] 모든 LUT perturbation (저주파+공분산+스무딩) 생성 완료")