
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval

# ---- 설정 ----
XMAX = 4095.0
NUM_LUTS = 350
GRAY_LEVELS = np.arange(0, XMAX + 1)
CONTROL_X = np.array([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4095], dtype=float)
SAVE_DIR = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\2. Monotonic\CSV"



# -----------------------------
# 1) 기존: 제약을 만족하는 offsets 생성
# -----------------------------
def generate_offsets_with_constraints(control_x):
    # control_x: [0,512,1024,1536,2048,2560,3072,3584,4095]
    idx_peak = 4
    n = len(control_x)
    offsets = np.zeros(n, dtype=float)

    bounds = np.minimum(control_x, 4095 - control_x)

    offsets[0] = 0.0
    offsets[8] = 0.0

    # 512 in [256,512] (또는 bound 좁으면 bound로)
    lo_512, hi_512 = 256.0, min(512.0, bounds[1]) # bounds[1]은 LUT가 0~4095 범위를 벗어나지 않도록 제한
    offsets[1] = np.random.uniform(min(lo_512, hi_512), hi_512)

    # 2048의 피크값
    peak_upper = min(1800.0, bounds[idx_peak])
    peak_lower = max(offsets[1], 600.0)
    if peak_lower > peak_upper:
        peak_lower = min(peak_upper, offsets[1])
    peak = np.random.uniform(peak_lower, peak_upper)

    # 512→2048 단조 증가 분배
    inc_total = max(0.0, peak - offsets[1])
    w_inc = np.random.random(3); w_inc = w_inc / (w_inc.sum() or 1.0)
    inc_steps = inc_total * np.cumsum(w_inc)
    offsets[2] = offsets[1] + inc_steps[0]
    offsets[3] = offsets[1] + inc_steps[1]
    offsets[4] = offsets[1] + inc_steps[2]

    for i in [2,3,4]:
        offsets[i] = min(offsets[i], bounds[i])
        offsets[i] = max(offsets[i], offsets[i-1])  # 비감소

    # 2048→4095 단조 감소 분배 (마지막은 0)
    dec_total = offsets[4]
    w_dec = np.random.random(4); w_dec = w_dec / (w_dec.sum() or 1.0)
    dec_cum = dec_total * np.cumsum(w_dec)
    offsets[5] = max(offsets[4] - dec_cum[0], 0.0)
    offsets[6] = max(offsets[4] - dec_cum[1], 0.0)
    offsets[7] = max(offsets[4] - dec_cum[2], 0.0)
    offsets[8] = 0.0

    for i in [5,6,7]:
        offsets[i] = min(offsets[i], bounds[i])
        offsets[i] = min(offsets[i], offsets[i-1])  # 비증가

    offsets = np.clip(offsets, 0, bounds)

    # 구조 검증
    if not (256.0 - 1e-6 <= offsets[1] <= 512.0 + 1e-6):
        return None
    if not (offsets[1] <= offsets[2] <= offsets[3] <= offsets[4] + 1e-6):
        return None
    if not (offsets[4] >= offsets[5] >= offsets[6] >= offsets[7] >= offsets[8] - 1e-6):
        return None

    print("Generated offsets at control points:")
    for i, (x, off) in enumerate(zip(control_x, offsets)):
        print(f"  Point {i}: Gray Level = {x:.0f}, Offset = {off:.2f}")

    return offsets

# ----------------------------------------
# 2) Zigzag 방지 보정: Δoffset 클리핑 (선형 보간용)
# ----------------------------------------
def enforce_no_zigzag(offsets, control_x, eps=1.0):
    """
    선형보간에서 기울기 음수 방지:
      - Low:  Δoffset <= Δx
      - High: Δoffset >= -Δx
    ⇒  -Δx+eps ≤ Δoffset ≤ Δx-eps 로 클립
    그리고 512→2048 비감소, 2048→4095 비증가를 다시 보정.
    """
    off = offsets.copy()
    dx = np.diff(control_x)  # 전 구간 512, 마지막만 494
    delta = np.diff(off)

    # 구간별 허용 범위
    low_bound  = -dx + eps
    high_bound =  dx - eps
    delta_clipped = np.clip(delta, low_bound, high_bound)

    # 누적 재구성
    new_off = np.zeros_like(off)
    new_off[0] = off[0]
    for i in range(1, len(off)):
        new_off[i] = new_off[i-1] + delta_clipped[i-1]

    # 구조(봉우리) 재보정: 512→2048 비감소, 2048→4095 비증가
    # 상승 구간
    for i in [2,3,4]:
        new_off[i] = max(new_off[i], new_off[i-1])
    # 하강 구간
    for i in [6,7,8]:
        new_off[i] = min(new_off[i], new_off[i-1])
    # 2560 지점은 피크(2048) 이하로
    new_off[5] = min(new_off[5], new_off[4])

    # 경계/특수점 재설정
    bounds = np.minimum(control_x, 4095 - control_x)
    new_off = np.clip(new_off, 0, bounds)
    new_off[0] = 0.0
    new_off[-1] = 0.0
    new_off[1] = np.clip(new_off[1], 256.0, min(512.0, bounds[1]))

    return new_off

# ----------------------------------------
# 3) LUT 생성(선형 보간만) + 플롯
# ----------------------------------------
def generate_linear_LUTs_no_zigzag(gray_levels=4096):
    control_x = np.array([0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4095], dtype=float)

    # 유효 offsets 생성
    for _ in range(1000):
        offsets = generate_offsets_with_constraints(control_x)
        if offsets is not None:
            break
    else:
        raise RuntimeError("Failed to sample offsets.")

    # Zigzag 방지 보정
    offsets = enforce_no_zigzag(offsets, control_x, eps=1.0)

    # Low / High 제어점
    cy_low  = control_x - offsets
    cy_high = control_x + offsets

    # 선형 보간 LUT
    x_full = np.arange(gray_levels)
    lut_low  = np.interp(x_full, control_x, cy_low)
    lut_high = np.interp(x_full, control_x, cy_high)

    # 정수 클립
    lut_low  = np.clip(lut_low,  0, 4095).astype(int)
    lut_high = np.clip(lut_high, 0, 4095).astype(int)

    return control_x.astype(int), cy_low.astype(int), cy_high.astype(int), lut_low, lut_high, offsets


# -----------------------------
# 다항식 피팅 함수
# -----------------------------
def fit_endpoint_constrained_poly(cx, cy, deg=4):
    """
    엔드포인트 고정: y(0)=0, y(XMAX)=XMAX 를 보장하는 형태
    y(x) = x + w(x) * sum_{k=0..deg} a_k T_k(t),  w(x)=x*(XMAX-x),  t = 2x/XMAX - 1
    -> a_k를 최소자승으로 추정
    """
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)

    # 타깃을 y - x 로 옮겨놓고, 가중 w(x)로 나눌 수 없으니 디자인행렬에 w(x)를 곱해준다.
    # 목표:  (y - x) ≈ w(x) * sum a_k T_k(t)
    t = 2.0*cx/XMAX - 1.0               # [-1,1]로 정규화된 x
    w = cx*(XMAX - cx)                  # 엔드포인트 제약을 보장하는 가중
    rhs = cy - cx                       # 타깃 벡터

    # 디자인 행렬: Phi[i,k] = w(x_i) * T_k(t_i)
    Phi = np.zeros((len(cx), deg+1), dtype=float)
    # 수치안정: chebval은 T_0..T_deg 값을 한 번에 계산하기 어려우니 반복
    for k in range(deg+1):
        # T_k(t) 값
        Tk = np.cos(k*np.arccos(np.clip(t, -1, 1)))  # 체비셰프 정의 이용
        Phi[:, k] = w * Tk

    # 최소자승 해 a = argmin ||Phi a - rhs||_2
    a, *_ = np.linalg.lstsq(Phi, rhs, rcond=None)

    def y_poly(x):
        x = np.asarray(x, dtype=float)
        tt = 2.0*x/XMAX - 1.0
        ww = x*(XMAX - x)
        # sum a_k T_k(tt)
        # chebval로 한 번에: chebval(tt, a) = sum a_k T_k(tt)
        q = chebval(tt, a)
        return x + ww * q

    return a, y_poly

# -----------------------------
# CSV 저장 함수
# -----------------------------
def save_lut_to_csv(filename, gray_levels, lut_low, lut_high):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 첫 번째 행: 제목
        writer.writerow(['Gray Level', 'R_Low', 'R_High', 'G_Low', 'G_High', 'B_Low', 'B_High'])
        # 데이터 행: R/G/B 동일하게 저장
        for g, low, high in zip(gray_levels, lut_low, lut_high):
            writer.writerow([g, low, high, low, high, low, high])    

# -----------------------------
# LUT 생성 및 저장
# -----------------------------
def generate_and_save_LUTs(start_idx=201, num_luts=100):
    for i in range(start_idx, start_idx + num_luts):
        # offsets 생성
        for _ in range(1000):
            offsets = generate_offsets_with_constraints(CONTROL_X)
            if offsets is not None:
                break
        else:
            raise RuntimeError(f"Failed to sample offsets for LUT {i}")
        
        offsets = enforce_no_zigzag(offsets, CONTROL_X)

        # 선형 LUT 생성
        cy_low  = CONTROL_X - offsets
        cy_high = CONTROL_X + offsets
        lut_low_linear  = np.interp(GRAY_LEVELS, CONTROL_X, cy_low)
        lut_high_linear = np.interp(GRAY_LEVELS, CONTROL_X, cy_high)
        lut_low_linear  = np.clip(np.round(lut_low_linear),  0, 4095).astype(int)
        lut_high_linear = np.clip(np.round(lut_high_linear), 0, 4095).astype(int)

        # 다항식 LUT 생성
        _, y_poly_low = fit_endpoint_constrained_poly(CONTROL_X, cy_low)
        _, y_poly_high = fit_endpoint_constrained_poly(CONTROL_X, cy_high)
        lut_low_poly  = np.clip(np.round(y_poly_low(GRAY_LEVELS)),  0, 4095).astype(int)
        lut_high_poly = np.clip(np.round(y_poly_high(GRAY_LEVELS)), 0, 4095).astype(int)
        
        # 파일 저장
        filename_linear = os.path.join(SAVE_DIR, f"LUT_Monotonic_Linear_{i}.csv")
        filename_poly   = os.path.join(SAVE_DIR, f"LUT_Monotonic_Polyfit_{i}.csv")
        
        save_lut_to_csv(filename_linear, GRAY_LEVELS, lut_low_linear, lut_high_linear)
        save_lut_to_csv(filename_poly, GRAY_LEVELS, lut_low_poly, lut_high_poly)



# -----------------------------
# 실행
# -----------------------------
generate_and_save_LUTs()





# # ===== 사용 예시: 기존 9포인트에서 다항식 피팅 =====
# cx, cy_low, cy_high, lut_low, lut_high, offsets = generate_linear_LUTs_no_zigzag()
# # cx, cy_low, cy_high 는 이전 셀(혹은 사용자 코드)에서 나온 9포인트를 사용
# # 예: cx, cy_low, cy_high = np.array([...]), np.array([...]), np.array([...])

# deg = 4  # 필요시 3~6 사이로 바꿔 보세요
# a_low,  y_low_poly  = fit_endpoint_constrained_poly(cx, cy_low,  deg=deg)
# a_high, y_high_poly = fit_endpoint_constrained_poly(cx, cy_high, deg=deg)

# # 곡선 샘플링 & 플롯
# x_full = np.arange(0, int(XMAX)+1)
# y_low_fit  = np.clip(y_low_poly(x_full),  0, XMAX)
# y_high_fit = np.clip(y_high_poly(x_full), 0, XMAX)

# plt.figure(figsize=(10,6))
# plt.plot(x_full, x_full, '--', color='gray', label='y = x')
# # plt.plot(x_full, y_low_fit,  label=f'Low fit (deg={deg})')
# # plt.plot(x_full, y_high_fit, label=f'High fit (deg={deg})')

# plt.plot(x_full, lut_low,  label='Linear Low LUT', linestyle='--')
# plt.plot(x_full, lut_high, label='Linear High LUT', linestyle='--')
# plt.plot(x_full, y_low_fit,  label='Polyfit Low LUT', linewidth=2)
# plt.plot(x_full, y_high_fit, label='Polyfit High LUT', linewidth=2)


# # 원래 9포인트 표시
# plt.scatter(cx, cy_low,  label='Low 9pts',  zorder=3)
# plt.scatter(cx, cy_high, label='High 9pts', zorder=3)

# plt.title('Endpoint-constrained Chebyshev polynomial fitting')
# plt.xlabel('Input Gray Level'); plt.ylabel('Output Gray Level')
# plt.grid(True); plt.legend(); plt.tight_layout()
# plt.show()

# # === 계수 확인 (사실상 함수식) ===
# print("Low-fit Chebyshev coeffs a_k:", a_low)
# print("High-fit Chebyshev coeffs a_k:", a_high)
# print("Model: y(x) = x + x*(4095-x) * sum_k a_k * T_k( 2x/4095 - 1 )")

