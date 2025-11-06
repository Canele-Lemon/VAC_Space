def debug_dump_delta_with_mapping(self, pk=None, ref_vac_set_pk: int = 1, verbose_lut: bool = False):
    # pk 지정되면 세트 PK 교체
    if pk is not None:
        self.PK = int(pk)

    # ΔLUT + 메타 + 매핑 먼저 불러오기
    pack = self.prepare_X_delta_raw_with_mapping(ref_vac_set_pk=ref_vac_set_pk)
    delta = pack["lut_delta_raw"]; meta = pack["meta"]; j_map = pack["mapping_j"]

    print(f"\n[DEBUG] ΔLUT(raw, target−ref @ VAC_SET_Info.PK={ref_vac_set_pk}) "
          f"@ mapped indices for VAC_SET_Info.PK={self.PK}")
    print("[META]")
    print(f"  panel_maker one-hot: {meta['panel_maker']}")
    print(f"  frame_rate         : {meta['frame_rate']}")
    print(f"  model_year         : {meta['model_year']}")
    print("\n[MAPPING] j[0..10] =", j_map[:11].tolist(), "...")

    # ---- 원본 4096 LUT도 같이 로딩 (세트 PK 기준) ----
    lut4096_target = self._load_vacdata_lut4096(self.PK)
    lut4096_ref    = self._load_vacdata_lut4096(ref_vac_set_pk)

    channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']
    for ch in channels:
        arr = delta[ch]
        print(f"\n--- {ch} ---  shape={arr.shape}, dtype={arr.dtype}")
        for g in (0,1,32,128,255):
            if 0 <= g < len(arr):
                j = int(j_map[g])
                print(f"  gray {g:3d} @ j={j:4d} : Δ={float(arr[g]): .3f}")

                if verbose_lut:
                    tgt_val = float(lut4096_target[ch][j])
                    ref_val = float(lut4096_ref[ch][j])
                    diff    = tgt_val - ref_val
                    print(
                        f"      target[{ch}][{j}]={tgt_val: .3f}, "
                        f"ref[{ch}][{j}]={ref_val: .3f}, "
                        f"target - ref={diff: .3f}"
                    )