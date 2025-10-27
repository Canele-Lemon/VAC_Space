def debug_dump_delta_example(target_pk=2444):
    print(f"[DEBUG] Checking ΔLUT for target PK={target_pk}")

    builder = VACInputBuilder(target_pk)

    # 절대 LUT (target 자체)
    X_abs = builder.prepare_X0()

    # ΔLUT = target - ref(PK=1)
    X_delta = builder.prepare_X_delta()

    lut_abs   = X_abs["lut"]
    lut_delta = X_delta["lut"]
    meta      = X_delta["meta"]  # meta는 target 기준으로 동일하므로 어느쪽을 써도 같아야 함

    print("\n[META INFO]")
    print(f"panel_maker one-hot : {meta['panel_maker']}")
    print(f"frame_rate          : {meta['frame_rate']}")
    print(f"model_year          : {meta['model_year']}")

    # 채널 리스트 (prepare_X0와 동일)
    channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']

    # 우리가 직접 눈으로 확인해볼 gray 인덱스 몇 개
    sample_grays = [0, 1, 32, 128, 255]

    for ch in channels:
        arr_abs   = lut_abs[ch]    # (256,) 절대 LUT 값 [0..1] 정규화
        arr_delta = lut_delta[ch]  # (256,) ΔLUT 값 = target - ref

        print(f"\n--- Channel: {ch} ---")
        print(f"  shape abs   : {arr_abs.shape}, dtype={arr_abs.dtype}")
        print(f"  shape delta : {arr_delta.shape}, dtype={arr_delta.dtype}")

        for g in sample_grays:
            if g >= len(arr_abs):
                continue
            v_abs   = float(arr_abs[g])
            v_delta = float(arr_delta[g])
            print(f"    gray {g:3d} : abs={v_abs: .6f} , delta={v_delta: .6f}")

    # 간단 sanity check:
    # ΔLUT가 전부 0에 가깝다면 → target LUT가 ref(LUT@PK=1)랑 거의 동일하다는 뜻
    # ΔLUT가 + 쪽이면 → target이 ref보다 더 크게 올려놓은 구간
    # ΔLUT가 - 쪽이면 → target이 ref보다 더 낮춘 구간
    print("\n[NOTE] If delta≈0 for all channels, target LUT is basically same as ref(PK=1).")
    print("[NOTE] Positive delta means target LUT is higher than ref at that gray, negative means lower.\n")


if __name__ == "__main__":
    debug_dump_delta_example(target_pk=2444)