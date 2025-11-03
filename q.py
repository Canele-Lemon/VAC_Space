def debug_dump_delta_with_mapping(self, pk=None, ref_vac_info_pk: int = 1):
    if pk is not None:
        self.PK = int(pk)

    pack = self.prepare_X_delta_raw_with_mapping(ref_vac_info_pk=ref_vac_info_pk)
    delta = pack["lut_delta_raw"]; meta = pack["meta"]; j_map = pack["mapping_j"]

    print(f"\n[DEBUG] ΔLUT(raw, target−ref@PK={ref_vac_info_pk}) @ mapped indices for PK={self.PK}")
    print("[META]")
    print(f"  panel_maker one-hot: {meta['panel_maker']}")
    print(f"  frame_rate         : {meta['frame_rate']}")
    print(f"  model_year         : {meta['model_year']}")
    print("\n[MAPPING] j[0..10] =", j_map[:11].tolist(), "...")

    channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']
    for ch in channels:
        arr = delta[ch]
        print(f"\n--- {ch} ---  shape={arr.shape}, dtype={arr.dtype}")
        for g in (0,1,32,128,255):
            if 0 <= g < len(arr):
                print(f"  gray {g:3d} @ j={int(j_map[g]):4d} : Δ={float(arr[g]): .3f}")