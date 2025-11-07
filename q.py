def compute_Y0_struct(self, patterns=('W', 'R', 'G', 'B')):
    """
    Y[0] detailed: 패턴별(W/R/G/B) 정면 Gamma 특성 차이 w/ self.ref_pk (dGamma, dCx, dCy)
    Gamma(g) = log(nor.Lv_g) / log(gray_norm_g)
    - gray_norm = gray/255
    - gray=0 → NaN
    - gray=255 → NaN
    - nor.Lv=0 → NaN        

    patterns:
        사용할 패턴 튜플. 예)
          ('W',)           -> W만
          ('W','R')        -> W, R만
          ('W','R','G','B')-> 기존과 동일

    return:
        {
          'W': {'dGamma': (256,), 'dCx': (256,), 'dCy': (256,)},
          'R': {...},
          'G': {...},
          'B': {...}
        }
        (요청한 patterns에 해당하는 키만 포함)
    """
    # 유효 패턴만 필터링
    all_pattern_map = {
        'W': "VAC_Gamma_W_Gray____",
        'R': "VAC_Gamma_R_Gray____",
        'G': "VAC_Gamma_G_Gray____",
        'B': "VAC_Gamma_B_Gray____",
    }
    # 사용자가 요청한 패턴 중 실제로 정의된 것만 사용
    patterns = tuple(p for p in patterns if p in all_pattern_map)
    if not patterns:
        logging.warning("[Y0] No valid patterns requested, returning empty dict.")
        return {}

    # 이 패턴들에 해당하는 Parameter 목록 구성
    parameters = [all_pattern_map[p] for p in patterns]
    components = ('Lv', 'Cx', 'Cy')
    L = 256

    # 타겟 / 레퍼런스 데이터 로드 (요청한 패턴에 대해서만)
    df_target = self._load_measure_data(self.pk, parameters=parameters, components=components)
    df_ref    = self._load_measure_data(self.ref_pk, parameters=parameters, components=components)

    if df_target.empty or df_ref.empty:
        logging.warning(f"[Y0] Missing data (PK={self.pk}, Ref={self.ref_pk})")
        # 요청한 patterns에 대해서만 zero 구조 반환
        return {
            p: {k: np.zeros(L, np.float32) for k in ('dGamma', 'dCx', 'dCy')}
            for p in patterns
        }

    def calc_gamma_array(df_lv_pattern: pd.DataFrame) -> np.ndarray:
        """
        nor.Lv = 0 또는 gray=0/255이면 NaN으로 남김
        """
        gamma = np.full(L, np.nan, dtype=np.float32)
        if not df_lv_pattern.empty:
            lv_dict = dict(
                zip(
                    df_lv_pattern['Gray_Level'].to_numpy(),
                    df_lv_pattern['Data'].to_numpy(dtype=np.float32)
                )
            )
            gray = np.arange(L, dtype=np.float32)
            gray_norm = gray / 255.0
            lv_norm = np.array(
                [lv_dict.get(int(g), np.nan) for g in gray],
                dtype=np.float32
            )

            # 계산 불가능 조건 마스크
            invalid_mask = (
                (gray == 0) | (gray == 255) |
                (lv_norm <= 0) | np.isnan(lv_norm)
            )

            # 로그 계산
            with np.errstate(divide='ignore', invalid='ignore'):
                log_lv = np.log(lv_norm)
                log_gray = np.log(gray_norm)
                gamma_vals = log_lv / log_gray

            gamma[~invalid_mask] = gamma_vals[~invalid_mask]
            gamma[invalid_mask] = np.nan  # 명시적으로 NaN 처리
        return gamma

    y0 = {}
    for ptn in patterns:
        # Lv
        lv_t = df_target[
            (df_target['Pattern_Window'] == ptn) &
            (df_target['Component'] == 'Lv')
        ]
        lv_r = df_ref[
            (df_ref['Pattern_Window'] == ptn) &
            (df_ref['Component'] == 'Lv')
        ]

        gamma_t = calc_gamma_array(lv_t)
        gamma_r = calc_gamma_array(lv_r)
        dGamma  = (gamma_t - gamma_r).astype(np.float32)

        def diff_component(comp: str) -> np.ndarray:
            arr = np.zeros(L, np.float32)
            sub_t = df_target[
                (df_target['Pattern_Window'] == ptn) &
                (df_target['Component'] == comp)
            ]
            sub_r = df_ref[
                (df_ref['Pattern_Window'] == ptn) &
                (df_ref['Component'] == comp)
            ]
            if not sub_t.empty and not sub_r.empty:
                t = sub_t.sort_values('Gray_Level')[['Gray_Level', 'Data']].to_numpy()
                r = sub_r.sort_values('Gray_Level')[['Gray_Level', 'Data']].to_numpy()
                map_t = dict(zip(t[:, 0].astype(int), t[:, 1].astype(np.float32)))
                map_r = dict(zip(r[:, 0].astype(int), r[:, 1].astype(np.float32)))
                for g in range(L):
                    vt = map_t.get(g, np.nan)
                    vr = map_r.get(g, np.nan)
                    arr[g] = vt - vr if not (np.isnan(vt) or np.isnan(vr)) else np.nan
            else:
                arr[:] = np.nan
            return arr

        dCx = diff_component('Cx').astype(np.float32)
        dCy = diff_component('Cy').astype(np.float32)

        y0[ptn] = {
            'dGamma': dGamma,
            'dCx': dCx,
            'dCy': dCy
        }

    return y0