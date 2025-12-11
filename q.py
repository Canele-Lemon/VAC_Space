    def debug_dump_delta_with_mapping(
        self,
        pk: int | None = None,
        ref_pk: int = 1,
        verbose_lut: bool = False,
        preview_grays: list[int] | None = None,
    ):
        """
        ΔLUT(raw, target−ref @ LUT_index_mapping) 를
        '데이터셋 느낌'으로 요약해서 출력하는 디버그용 메서드.

        Parameters
        ----------
        pk : int | None
            target VAC_SET_Info PK (None이면 self.pk 그대로 사용)
        ref_pk : int
            reference VAC_SET_Info PK (보통 bypass LUT 세트 PK)
        verbose_lut : bool
            True이면 target/ref 원본 LUT 값도 함께 출력 (상세 검증용)
        preview_grays : list[int] | None
            테이블로 미리보기할 gray 인덱스 리스트.
            None이면 [0, 1, 32, 128, 255]를 기본 사용.
        """
        import pandas as pd
        import numpy as np

        # pk 지정되면 세트 PK 교체
        if pk is not None:
            self.pk = int(pk)

        # 1) ΔLUT + 메타 + 매핑 먼저 생성
        pack = self.prepare_X_delta_lut_with_mapping(ref_pk=ref_pk)
        delta = pack["lut_delta_raw"]   # dict[ch] = (256,) float32
        meta  = pack["meta"]            # panel_maker, frame_rate, model_year
        j_map = pack["mapping_j"]       # (256,) int32

        print(f"\n[DEBUG] ΔLUT(raw, target−ref @ VAC_SET_Info.PK={ref_pk}) "
              f"@ mapped indices for VAC_SET_Info.PK={self.pk}")

        # 2) META 요약
        print("\n[META]")
        print(f"  panel_maker one-hot: {meta['panel_maker']}")
        print(f"  frame_rate         : {meta['frame_rate']}")
        print(f"  model_year         : {meta['model_year']}")

        # 3) MAPPING 요약
        print("\n[MAPPING] j[0..10] =", j_map[:11].tolist(), "...")
        print(f"  mapping_j shape = {j_map.shape}, dtype={j_map.dtype}")

        # 4) 각 채널별 shape 정보
        channels = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']
        print("\n[ΔLUT SHAPES]")
        for ch in channels:
            arr = delta[ch]
            print(f"  {ch:7s}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={np.nanmin(arr): .1f}, max={np.nanmax(arr): .1f}")

        # 5) Dataset 스타일 미리보기 (몇 개 gray에 대해 행 구성)
        if preview_grays is None:
            preview_grays = [0, 1, 32, 128, 255]

        rows = []
        for g in preview_grays:
            if not (0 <= g < len(j_map)):
                continue
            j = int(j_map[g])
            row = {
                "gray": g,
                "LUT_j": j,
            }
            for ch in channels:
                row[f"d{ch}"] = float(delta[ch][g])
            rows.append(row)

        if rows:
            df_preview = pd.DataFrame(rows)
            print("\n[PREVIEW] ΔLUT per gray (깊이 있는 Dataset 느낌) :")
            print(df_preview.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
        else:
            print("\n[PREVIEW] no valid rows for preview_grays =", preview_grays)

        # 6) 원본 4096 LUT와의 일치 여부를 점검하고 싶을 때 (옵션)
        if verbose_lut:
            print("\n[VERBOSE] 타깃/레퍼런스 원본 LUT 값까지 검증:")

            lut4096_target = self._load_vacdata_lut4096(self.pk)
            lut4096_ref    = self._load_vacdata_lut4096(ref_pk)

            for ch in channels:
                print(f"\n--- {ch} ---")
                arr = delta[ch]
                for g in preview_grays:
                    if 0 <= g < len(arr):
                        j = int(j_map[g])
                        tgt_val = float(lut4096_target[ch][j])
                        ref_val = float(lut4096_ref[ch][j])
                        diff    = tgt_val - ref_val
                        print(
                            f"  gray {g:3d} @ j={j:4d} : "
                            f"Δ(from pack)={float(arr[g]): .3f}, "
                            f"target={tgt_val: .3f}, ref={ref_val: .3f}, "
                            f"target-ref={diff: .3f}"
                        )