    def compute_Y1_struct(self, patterns=('W',)):
        """
        Y[1] detailed: 측면 중계조 표현력 요약

        정의:
        - 측면(예: 60도) Nor.Lv 곡선에서
          gray=88 ~ 232 구간을 8-gray 간격으로 나눔
          (88–96, 96–104, ..., 224–232 → 총 18개 구간)
        - 각 구간에 대해 Nor.Lv vs gray/255 의 기울기
          slope = ΔNor.Lv / Δ(gray/255) 를 계산

        return:
            {
              'W': (18,),
              'R': (18,),
              'G': (18,),
              'B': (18,)
            }
        """

        # 60도 GammaLinearity 측정 파라미터 정의 (패턴별)
        all_params = {
            "W": "VAC_GammaLinearity_60_W_Gray____",
            "R": "VAC_GammaLinearity_60_R_Gray____",
            "G": "VAC_GammaLinearity_60_G_Gray____",
            "B": "VAC_GammaLinearity_60_B_Gray____",
        }
        # 요청된 패턴만 필터
        patterns = tuple(p for p in patterns if p in all_params)
        if not patterns:
            logging.warning("[Y1] No valid patterns requested, returning empty dict.")
            return {}

        parameters = [all_params[p] for p in patterns]

        # Nor.Lv 정규화까지 포함해서 불러온다 (normalize_lv_flag=True 기본값 유지)
        df = self._load_measure_data(
            self.pk,
            components=('Lv',),
            parameters=parameters,
        )

        L = 256
        y1 = {}

        # 구간 설정: 88~232, step=8 → 88,96,...,224 (마지막은 224→232)
        g_start = 88
        g_end   = 232
        step    = 8
        seg_starts = list(range(g_start, g_end, step))  # [88,96,...,224]

        for ptn in patterns:
            sub = df[(df['Pattern_Window'] == ptn) & (df['Component'] == 'Lv')].copy()
            sub = sub.sort_values('Gray_Level')

            # Gray_Level → Nor.Lv 매핑
            lv_dict = dict(
                zip(
                    sub['Gray_Level'].astype(int).to_numpy(),
                    sub['Data'].astype(float).to_numpy()
                )
            )

            slopes = []
            for gs in seg_starts:
                ge = gs + step
                if ge > 255:
                    continue  # 안전장치

                lv_s = lv_dict.get(gs, np.nan)
                lv_e = lv_dict.get(ge, np.nan)

                if not np.isfinite(lv_s) or not np.isfinite(lv_e):
                    slope = np.nan
                else:
                    gray_s = gs / 255.0
                    gray_e = ge / 255.0
                    denom = gray_e - gray_s
                    if denom <= 0:
                        slope = np.nan
                    else:
                        # ΔNor.Lv / Δ(gray/255)
                        slope = (lv_e - lv_s) / denom

                slopes.append(slope)

            if not slopes:
                # 데이터가 너무 부족한 경우 fallback
                y1[ptn] = np.full(len(seg_starts), np.nan, dtype=np.float32)
            else:
                y1[ptn] = np.asarray(slopes, dtype=np.float32)

        return y1