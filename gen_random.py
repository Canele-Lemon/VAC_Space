import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial

# 경로 설정
ref_dir = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\3. Perturbation\Ref"
output_dir = r"D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\Gen VAC\Random VAC\3. Perturbation\CSV\임시"
os.makedirs(output_dir, exist_ok=True)

# Perturbation 설정
LOW_RATE = 0.10   # 저계조
MID_RATE = 0.05   # 중계조
HIGH_RATE = 0.02  # 고계조

# 지그재그 보정 함수
def smooth_curve(values, max_iter=10):
    smoothed = values.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(1, len(smoothed) - 1):
            prev_grad = smoothed[i] - smoothed[i - 1]
            next_grad = smoothed[i + 1] - smoothed[i]
            if prev_grad * next_grad < 0:
                smoothed[i] = int(round((smoothed[i - 1] + smoothed[i + 1]) / 2))
                changed = True
        if not changed:
            break
    return smoothed

NUM_SAMPLE = 100
# Ref 폴더 내 모든 CSV 파일 처리
for file_name in os.listdir(ref_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(ref_dir, file_name)
        df = pd.read_csv(file_path, sep="\t" if "\t" in open(file_path).readline() else ",")

        # Gray Level (0~4095)
        gray_levels = df['Gray Level'].values
        base_values = df.drop(columns=['Gray Level']).values

        # Perturbation 대상 인덱스: 0 제외, 4095 제외, 16 간격
        perturbed_idx = np.arange(16, 4080 + 1, 16)
        fixed_idx = np.array([0, 4095])
        all_idx = np.concatenate([fixed_idx, perturbed_idx])
        all_idx.sort()

        sampled_gray = gray_levels[all_idx]
        sampled_values = base_values[all_idx]

        # for sample_num in range(1, NUM_SAMPLE + 1):    
        for sample_num in range(101, 101 + NUM_SAMPLE):

            # 각 커브에 대해 방향 결정
            directions = {}
            for ch in ['R', 'G', 'B']:
                for typ in ['Low', 'High']:
                    col_name = f"{ch}_{typ}"
                    directions[col_name] = np.random.choice([-1, 1])  # + 또는 - 방향 선택

            # Perturbation 적용
            perturbed_256 = sampled_values.copy()

            for ch in ['R', 'G', 'B']:
                for typ in ['Low', 'High']:
                    col_name = f"{ch}_{typ}"
                    col_idx = df.columns.get_loc(col_name) - 1
                    direction = directions[col_name]

                    for idx in range(1, len(sampled_gray) - 1):  # exclude 0 and 4095
                        gray = sampled_gray[idx]

                        # 계조별 rate 선택
                        if gray <= 1023:
                            rate = LOW_RATE
                        elif gray <= 3071:
                            rate = MID_RATE
                        else:
                            rate = HIGH_RATE

                        factor = 1 + direction * np.random.uniform(0, rate)
                        new_value = sampled_values[idx, col_idx] * factor
                        # perturbed_256[idx, col_idx] = max(0, int(round(new_value)))
                        perturbed_256[idx, col_idx] = min(4095, max(0, int(round(new_value))))


            # Low < High 조건 보정
            for ch in ['R', 'G', 'B']:
                low_idx = df.columns.get_loc(f"{ch}_Low") - 1
                high_idx = df.columns.get_loc(f"{ch}_High") - 1
                for idx in range(len(perturbed_256)):
                    if perturbed_256[idx, low_idx] >= perturbed_256[idx, high_idx]:
                        perturbed_256[idx, high_idx] = perturbed_256[idx, low_idx] + 1

            # 지그재그 보정
            for ch in ['R', 'G', 'B']:
                for typ in ['Low', 'High']:
                    col_name = f"{ch}_{typ}"
                    col_idx = df.columns.get_loc(col_name) - 1
                    perturbed_256[:, col_idx] = smooth_curve(perturbed_256[:, col_idx])

            # 선형 보간으로 전체 4096 포인트 생성
            full_gray = np.arange(0, 4096)
            full_lut = np.zeros((4096, base_values.shape[1]), dtype=int)

            for ch in ['R', 'G', 'B']:
                for typ in ['Low', 'High']:
                    col_name = f"{ch}_{typ}"
                    col_idx = df.columns.get_loc(col_name) - 1
                    interp_func = interp1d(sampled_gray, perturbed_256[:, col_idx], kind='linear', fill_value="extrapolate")
                    # full_lut[:, col_idx] = np.round(interp_func(full_gray)).astype(int)
                    full_lut[:, col_idx] = np.clip(np.round(interp_func(full_gray)).astype(int), 0, 4095)
 

            # CSV 저장
            ref_base_name = os.path.splitext(file_name)[0]
            out_file_name = f"LUT_Perturbation_{ref_base_name}_{sample_num}.csv"
            out_path = os.path.join(output_dir, out_file_name)

            out_df = pd.DataFrame(full_lut, columns=df.columns[1:])
            out_df.insert(0, 'Gray Level', full_gray)
            out_df.to_csv(out_path, index=False)

            # # 그래프 출력
            # plt.figure(figsize=(14, 6))
            # for ch in ['R', 'G', 'B']:
            #     for typ in ['Low', 'High']:
            #         col_name = f"{ch}_{typ}"
            #         col_idx = df.columns.get_loc(col_name) - 1

            #         # 원본 Ref
            #         ref_curve = base_values[:, col_idx]
            #         plt.plot(gray_levels, ref_curve, label=f"{col_name} Ref", linestyle='dotted', alpha=0.6)

            #         # 선형 보간
            #         linear_interp = interp1d(sampled_gray, perturbed_256[:, col_idx], kind='linear', fill_value="extrapolate")
            #         linear_curve = linear_interp(full_gray)
            #         plt.plot(full_gray, linear_curve, label=f"{col_name} Linear", linestyle='--')

            #         # Perturbation 포인트 표시
            #         plt.scatter(sampled_gray[1:-1], perturbed_256[1:-1, col_idx], label=f"{col_name} Perturbed", marker='o', s=20)

            # plt.title(f"Perturbation & Interpolation: {file_name}")
            # plt.xlabel("Gray Level (12bit)")
            # plt.ylabel("Signal Value")
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

print("[✅] 모든 LUT Perturbation 완료")
