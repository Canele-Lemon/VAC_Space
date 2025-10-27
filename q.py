import os, sys

# 현재 파일 경로 기준으로 프로젝트 루트(= module 디렉토리)를 sys.path에 추가
# ex) module/src/prepare_output.py -> module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../module/src
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) # .../module
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    
def debug_dump_y0_delta(target_pk=2444, reference_pk=2154, preview_grays=(0, 1, 32, 128, 255)):
    """
    1) target_pk와 reference_pk 각각의 raw 측정 데이터를 CSV로 저장하고 엽니다.
    2) 두 데이터로부터 계산된 dGamma / dCx / dCy 일부 값을 콘솔에 출력합니다.
    """
    print(f"[DEBUG] target_pk={target_pk}, reference_pk={reference_pk}")

    # 빌더 준비
    target_builder = VACOutputBuilder(pk=target_pk, reference_pk=reference_pk)
    ref_builder    = VACOutputBuilder(pk=reference_pk, reference_pk=reference_pk)

    # 1) RAW MEASUREMENTS 덤프 → CSV
    print("\n[STEP 1] Dump raw measurement rows for target and reference to CSV (and open)")
    df_target = target_builder.load_set_info_pk_data(target_pk)
    df_ref    = ref_builder.load_set_info_pk_data(reference_pk)

    print(f"  target df rows: {len(df_target)} (pk={target_pk})")
    print(f"  ref    df rows: {len(df_ref)}    (pk={reference_pk})")

    # 2) ΔY0 계산 (dGamma, dCx, dCy)
    print("\n[STEP 2] Compute dGamma / dCx / dCy using compute_Y0_struct() ...")
    y0_delta = target_builder.compute_Y0_struct()

    # y0_delta 구조 예:
    # {
    #   'W': {'dGamma': (256,), 'dCx': (256,), 'dCy': (256,)},
    #   'R': {...},
    #   'G': {...},
    #   'B': {...}
    # }

    patterns = ['W','R','G','B']
    for ptn in patterns:
        if ptn not in y0_delta:
            continue

        dGamma_arr = y0_delta[ptn]['dGamma']
        dCx_arr    = y0_delta[ptn]['dCx']
        dCy_arr    = y0_delta[ptn]['dCy']

        print(f"\n[PTN {ptn}]  (values are target - ref)")
        for g in preview_grays:
            if g < 0 or g >= len(dGamma_arr):
                continue

            dGamma_val = float(dGamma_arr[g]) if np.isfinite(dGamma_arr[g]) else None
            dCx_val    = float(dCx_arr[g])    if np.isfinite(dCx_arr[g])    else None
            dCy_val    = float(dCy_arr[g])    if np.isfinite(dCy_arr[g])    else None

            print(f"  gray {g:3d}:  dGamma={dGamma_val} , dCx={dCx_val} , dCy={dCy_val}")

    print("\n[STEP 3] Sanity checklist:")
    print(" - 위 dCx, dCy는 실제로 df_target에서 해당 gray의 Cx/Cy 값 minus df_ref의 Cx/Cy 값과 일치해야 합니다.")
    print(" - dGamma는 정규화된 Lv → gamma 계산 후 차이이므로, 그냥 Lv 차이랑은 다를 수 있습니다.")
    print(" - dGamma/dCx/dCy가 거의 0이라면 target_pk와 reference_pk의 특성이 매우 유사하다는 뜻입니다.")
    
