    def debug_Y0_dataset(
        self,
        patterns=('W', 'R', 'G', 'B'),
        gray_samples=None,
    ):
        """
        Y0 = (target - ref) 의 dGamma/dCx/dCy 를
        'per-gray dataset' 느낌으로 프리뷰하는 디버그용 메서드.

        parameters
        ----------
        patterns : tuple
            확인할 패턴 목록
        gray_samples : list[int] | None
            미리보기할 gray 인덱스 리스트 (None이면 [0,1,32,128,255])
        """
        if gray_samples is None:
            gray_samples = [0, 1, 32, 128, 255]

        print(f"\n[DEBUG Y0] pk={self.pk}, ref_pk={self.ref_pk}, patterns={patterns}")

        y0 = self.compute_Y0_struct(patterns=patterns)
        if not y0:
            print("[DEBUG Y0] empty Y0 struct (데이터 없음)")
            return

        for ptn in patterns:
            if ptn not in y0:
                continue

            dG = y0[ptn]['dGamma']  # (256,)
            dCx = y0[ptn]['dCx']
            dCy = y0[ptn]['dCy']

            print(f"\n[Pattern {ptn}] shapes: dGamma={dG.shape}, dCx={dCx.shape}, dCy={dCy.shape}")

            rows = []
            for g in gray_samples:
                if 0 <= g < len(dG):
                    rows.append({
                        "gray": g,
                        "dGamma": float(dG[g]),
                        "dCx": float(dCx[g]),
                        "dCy": float(dCy[g]),
                    })
            if rows:
                df_prev = pd.DataFrame(rows)
                print(df_prev.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
            else:
                print("  (no valid gray samples)")
                
    def debug_Y1_dataset(self, patterns=('W',), g_start=88, g_end=232, step=8):
        """
        Y1 = 측면 Nor.Lv slope 요약(88~232, interval 8) 을
        segment index / gray 구간과 함께 DataFrame으로 프리뷰.

        각 패턴에 대해:
        seg_idx, g_start, g_end, slope
        """
        print(f"\n[DEBUG Y1] pk={self.pk}, patterns={patterns}")
        y1 = self.compute_Y1_struct(patterns=patterns)
        if not y1:
            print("[DEBUG Y1] empty Y1 struct (데이터 없음)")
            return

        seg_starts = list(range(g_start, g_end, step))  # [88, 96, ..., 224]

        for ptn in patterns:
            if ptn not in y1:
                continue

            slopes = np.asarray(y1[ptn], dtype=np.float32)
            print(f"\n[Pattern {ptn}] slopes shape = {slopes.shape}")

            rows = []
            for idx, gs in enumerate(seg_starts):
                ge = gs + step
                if idx >= len(slopes):
                    break
                rows.append({
                    "seg_idx": idx,
                    "g_start": gs,
                    "g_end": ge,
                    "slope": float(slopes[idx]),
                })

            if rows:
                df_prev = pd.DataFrame(rows)
                print(df_prev.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
            else:
                print("  (no slope rows)")
                
    def debug_Y2_dataset(self):
        """
        Y2 (Macbeth 12패턴 Δu'v')를
        작은 테이블 형태로 프리뷰.
        """
        print(f"\n[DEBUG Y2] pk={self.pk}")
        y2 = self.compute_Y2_struct()
        if not y2:
            print("[DEBUG Y2] empty Y2 struct (데이터 없음)")
            return

        rows = []
        for name, val in y2.items():
            rows.append({"Macbeth": name, "delta_uv": float(val)})

        df_prev = pd.DataFrame(rows)
        print(df_prev.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
        
    def debug_full_Y_dataset(
        self,
        y0_patterns=('W', 'R', 'G', 'B'),
        y1_patterns=('W',),
        gray_samples_Y0=None,
        show_Y2=True,
    ):
        """
        Y0/Y1/Y2 전체를 'Dataset 느낌'으로 한 번에 프리뷰하는 헬퍼.

        - Y0: 패턴별 dGamma/dCx/dCy @ selected grays
        - Y1: 패턴별 88~232 slope segments
        - Y2: Macbeth delta_uv 테이블
        """
        print(f"\n============================")
        print(f"[DEBUG FULL Y] pk={self.pk}, ref_pk={self.ref_pk}")
        print(f"============================")

        # Y0
        self.debug_Y0_dataset(patterns=y0_patterns, gray_samples=gray_samples_Y0)

        # Y1
        self.debug_Y1_dataset(patterns=y1_patterns)

        # Y2
        if show_Y2:
            self.debug_Y2_dataset()