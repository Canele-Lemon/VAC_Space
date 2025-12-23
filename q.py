# prepare_output.py
import os
import sys
import logging
import tempfile
import webbrowser

import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    
from config.db_config import engine

logging.basicConfig(level=logging.DEBUG)

class VACOutputBuilder:
    def __init__(self, pk: int, ref_pk: int):
        self.pk = pk
        self.ref_pk = ref_pk
        
        self.MEASUREMENT_INFO_TABLE = "W_VAC_SET_Info"
        self.MEASUREMENT_DATA_TABLE = "W_VAC_SET_Measure"
        
    def _load_measure_data(self, pk, parameters=None, components=('Lv', 'Cx', 'Cy'), normalize_lv_flag=True):
        # [STEP 1] 데이터 로드
        parameter_condition = " OR ".join([f"`Parameter` LIKE '{p}'" for p in parameters])
        components_str = "(" + ", ".join([f"'{c}'" for c in components]) + ")"
        
        query = f"""
        SELECT `Parameter`, `Component`, `Data`
        FROM `{self.MEASUREMENT_DATA_TABLE}`
        WHERE `VAC_SET_Info_PK` = {pk}
        AND ({parameter_condition})
        AND `Component` IN {components_str}
        """
        df = pd.read_sql(query, engine)
        
        # logging.debug(f"[STEP 1] 측정 SET 정보 PK={pk}의 RAW DATA:\n{df.head()}")
        
        df['Data'] = df['Data'].astype(float)
        df['Pattern_Window'] = df['Parameter'].str.extract(
            r'(?:VAC_(?:Gamma(?:Linearity_60)?|ColorShift_\d{1,3})_)([A-Za-z ]+)'
        )
        if df['Parameter'].str.contains('VAC_Gamma|VAC_GammaLinearity').any():
            df['Gray_Level'] = df['Parameter'].str.extract(r'Gray(\d{4})')[0].fillna(-1).infer_objects(copy=False).astype(int)
            df = df[['Pattern_Window', 'Gray_Level', 'Component', 'Data']]
        else:
            df = df[['Pattern_Window', 'Component', 'Data']]

        # logging.debug(f"[STEP 1] 데이터 로드:\n{df.head(6)}")

        # [STEP 2] Lv 이상치 보정 (Lv@Gray0 > Lv@Gray1이면 Lv@Gray0 = Lv@Gray1, Cx@Gray0 = Cx@Gray1, Cy@Gray0 = Cy@Gray1)
        if 'Lv' in components and normalize_lv_flag:
            lv_df = df[df['Component'] == 'Lv']
            
            # Lv@Gray0, Lv@Gray1, Lv@Gray255 추출
            lv_0 = lv_df[lv_df['Gray_Level'] == 0].set_index('Pattern_Window')['Data'].to_dict()
            lv_1 = lv_df[lv_df['Gray_Level'] == 1].set_index('Pattern_Window')['Data'].to_dict()
            lv_255 = lv_df[lv_df['Gray_Level'] == 255].set_index('Pattern_Window')['Data'].to_dict()
            
            lv_0_fixed = {} # Lv@Gray0 보정값을 저장할 딕셔너리
            for PTN_win in lv_0:
                lv0 = lv_0[PTN_win]
                lv1 = lv_1.get(PTN_win, lv0)

                if lv0 > lv1:
                    fixed_lv0 = lv1
                    lv_0_fixed[PTN_win] = fixed_lv0

                    # Cx/Cy 보정: Gray0 값을 Gray1 값으로 덮어쓰기
                    for comp in ['Cx', 'Cy']:
                        cond_0 = (df['Pattern_Window'] == PTN_win) & (df['Gray_Level'] == 0) & (df['Component'] == comp)
                        cond_1 = (df['Pattern_Window'] == PTN_win) & (df['Gray_Level'] == 1) & (df['Component'] == comp)
                        val_1 = df.loc[cond_1, 'Data'].values
                        if len(val_1) > 0:
                            df.loc[cond_0, 'Data'] = val_1[0]

                else:
                    lv_0_fixed[PTN_win] = lv0  # 이상치가 아니면 원래 값 유지
                
            # Lv@Gray0 보정값을 df에 반영
            df.loc[(df['Component'] == 'Lv') & (df['Gray_Level'] == 0), 'Data'] = \
                df.loc[(df['Component'] == 'Lv') & (df['Gray_Level'] == 0), 'Pattern_Window'].map(lv_0_fixed)
            
            # logging.debug(f"[STEP 2] Lv 이상치 보정 적용 후 df:\n{df.head(6)}")

        # [STEP 3] Lv 정규화: (Lv - Lv@Gray0) / (Lv@Gray255 - Lv@Gray0)
        if 'Lv' in components and normalize_lv_flag:
            lv_df = df[df['Component'] == 'Lv'].copy()
            
            normalized_lv = []
            for pattern in lv_df['Pattern_Window'].unique():
                sub = lv_df[lv_df['Pattern_Window'] == pattern].copy()
                
                lv_min = sub['Data'].min()
                lv_max = sub['Data'].max()
                
                denom = lv_max - lv_min if lv_max != lv_min else 1.0
                
                sub['Data'] = (sub['Data'] - lv_min) / denom
                normalized_lv.append(sub)

            lv_df = pd.concat(normalized_lv, ignore_index=True)
            # logging.debug(f"[STEP 3] Lv 정규화 후 lv_df:\n{lv_df.head()}")

        df = df[df['Component'] != 'Lv'].copy() # 기존 df에서 Lv 데이터를 제거
        
        if 'lv_df' in locals():
            df = pd.concat([df, lv_df], ignore_index=True)

        # df = pd.concat([df, lv_df], ignore_index=True) # 정규화된 Lv 데이터를 병합
        
        # logging.debug(f"[STEP 3] Lv 정규화 후 df:\n{df.head(6)}")
        
        return df

    def compute_Y0_struct(self, patterns=('W',)):
        """
        Y[0] detailed: 패턴별(W/R/G/B) self.pk와 self.ref_pk 간 Gamma, Cx, Cy 차이 => dGamma, dCx, dCy
        Gamma(g) = log(nor.Lv_g) / log(gray_norm_g)
        - gray_norm = gray/255
        - gray=0 → NaN
        - gray=255 → NaN
        - nor.Lv=0 → NaN        
        
        return:
        {
          'W': {'dGamma': (256,), 'dCx': (256,), 'dCy': (256,)},
          'R': {...},
          'G': {...},
          'B': {...}
        } 
        ※ dGamma를 계산할 수 없는 경우 NaN으로 처리
        """
        all_pattern_map = {
            'W': "VAC_Gamma_W_Gray____",
            'R': "VAC_Gamma_R_Gray____",
            'G': "VAC_Gamma_G_Gray____",
            'B': "VAC_Gamma_B_Gray____",
        }
        patterns = tuple(p for p in patterns if p in all_pattern_map)
        if not patterns:
            logging.warning("[Y0] No valid patterns requested, returning empty dict.")
            return {}
 
        parameters = [all_pattern_map[p] for p in patterns]
        components = ('Lv', 'Cx', 'Cy')
        L = 256

        df_target = self._load_measure_data(self.pk, parameters=parameters, components=components)
        df_ref = self._load_measure_data(self.ref_pk, parameters=parameters, components=components)

        if df_target.empty or df_ref.empty:
            logging.warning(f"[Y0] Missing data (PK={self.pk}, Ref={self.ref_pk})")
            return {
                p: {k: np.zeros(L, np.float32) for k in ('dGamma', 'dCx', 'dCy')}
                for p in patterns
            }

        def calc_gamma_array(df_lv_pattern: pd.DataFrame) -> np.ndarray:
            """
            Gamma를 계산합니다.
            ※ nor.Lv = 0 또는 gray=0/255이면 NaN으로 처리
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
                lv_norm = np.array([lv_dict.get(int(g), np.nan) for g in gray], dtype=np.float32)

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
            lv_t = df_target[(df_target['Pattern_Window'] == ptn) & (df_target['Component'] == 'Lv')]
            lv_r = df_ref[(df_ref['Pattern_Window'] == ptn) & (df_ref['Component'] == 'Lv')]

            gamma_t = calc_gamma_array(lv_t)
            gamma_r = calc_gamma_array(lv_r)
            dGamma  = (gamma_t - gamma_r).astype(np.float32)

            def diff_component(comp: str) -> np.ndarray:
                arr = np.zeros(L, np.float32)
                sub_t = df_target[(df_target['Pattern_Window'] == ptn) & (df_target['Component'] == comp)]
                sub_r = df_ref[(df_ref['Pattern_Window'] == ptn) & (df_ref['Component'] == comp)]
                if not sub_t.empty and not sub_r.empty:
                    t = sub_t.sort_values('Gray_Level')[['Gray_Level','Data']].to_numpy()
                    r = sub_r.sort_values('Gray_Level')[['Gray_Level','Data']].to_numpy()
                    map_t = dict(zip(t[:,0].astype(int), t[:,1].astype(np.float32)))
                    map_r = dict(zip(r[:,0].astype(int), r[:,1].astype(np.float32)))
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
    
    def compute_Y0_struct_abs(self):
        """
        Y0_abs: 패턴별(W/R/G/B) 정면 절대 Gamma, Cx, Cy
        Gamma(g) = log(nor.Lv_g) / log(gray_norm_g)
        - gray=0/255 또는 nor.Lv<=0 → NaN 그대로 유지
        
        return:
        { 'W': {'Gamma': (256,), 'Cx': (256,), 'Cy': (256,)},
          'R': {...}, 
          'G': {...}, 
          'B': {...}
        }
        """
        parameters = [
            "VAC_Gamma_W_Gray____",
            "VAC_Gamma_R_Gray____",
            "VAC_Gamma_G_Gray____",
            "VAC_Gamma_B_Gray____"
        ]
        df = self._load_measure_data(self.pk, parameters=parameters, components=('Lv','Cx','Cy'))
        L, patterns = 256, ('W','R','G','B')

        def calc_gamma_array(df_lv_pattern):
            import numpy as np
            gamma = np.full(L, np.nan, np.float32)
            if not df_lv_pattern.empty:
                lv_dict = dict(zip(df_lv_pattern['Gray_Level'].to_numpy(),
                                df_lv_pattern['Data'].to_numpy(np.float32)))
                gray = np.arange(L, dtype=np.float32)
                gray_norm = gray/255.0
                lv_norm = np.array([lv_dict.get(int(g), np.nan) for g in gray], dtype=np.float32)
                invalid = (gray==0)|(gray==255)|(lv_norm<=0)|np.isnan(lv_norm)
                with np.errstate(divide='ignore', invalid='ignore'):
                    gamma_vals = np.log(lv_norm)/np.log(gray_norm)
                gamma[~invalid] = gamma_vals[~invalid]
            return gamma

        y0_abs = {}
        for p in patterns:
            lv_p = df[(df['Pattern_Window']==p)&(df['Component']=='Lv')]
            cx_p = df[(df['Pattern_Window']==p)&(df['Component']=='Cx')]
            cy_p = df[(df['Pattern_Window']==p)&(df['Component']=='Cy')]

            Gamma = calc_gamma_array(lv_p)

            Cx = np.full(L, np.nan, np.float32)
            Cy = np.full(L, np.nan, np.float32)
            if not cx_p.empty:
                for g,v in zip(cx_p['Gray_Level'], cx_p['Data']):
                    if 0<=g<L: Cx[int(g)] = np.float32(v)
            if not cy_p.empty:
                for g,v in zip(cy_p['Gray_Level'], cy_p['Data']):
                    if 0<=g<L: Cy[int(g)] = np.float32(v)

            y0_abs[p] = {'Gamma': Gamma, 'Cx': Cx, 'Cy': Cy}

        return y0_abs

    def compute_Y1_struct(self, patterns=('W',)):
        """
        Y[1] detailed: 측면 중계조 표현력 (Nor.Lv(gray_end) - Nor.Lv(gray_start)) / ((gray_end-gray_start)/255)
        
        return:
        *patterns=('W','R','G','B') 선택 시,
            {
              'W': (18,),
              'R': (18,),
              'G': (18,),
              'B': (18,)
            }
        """
        all_pattern_map = {
            'W': "VAC_GammaLinearity_60_W_Gray____",
            'R': "VAC_GammaLinearity_60_R_Gray____",
            'G': "VAC_GammaLinearity_60_G_Gray____",
            'B': "VAC_GammaLinearity_60_B_Gray____",
        }
        patterns = tuple(p for p in patterns if p in all_pattern_map)
        if not patterns:
            logging.warning("[Y1] No valid patterns requested, returning empty dict.")
            return {}
        
        parameters = [all_pattern_map[p] for p in patterns]
        
        df = self._load_measure_data(self.pk, components=('Lv',), parameters=parameters)
        
        L = 256
        y1 = {}
        
        # 구간 설정 (18구간)
        # 88-96, 96-104, 104-112, 112-120, 120-128, 128-136, 136-144, 144-152, 152-160, 160-168,
        # 168-176, 176-184, 184-192, 192-200, 200-208, 208-216, 216-224, 224-232 
        g_start = 88
        g_end = 232
        step = 8
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
                arr = np.asarray(slopes, dtype=np.float32)
                arr = np.abs(arr) # 절대값 처리
                y1[ptn] = arr

        return y1

    def compute_Y2_struct(self):
        """
        Y[2] detailed: Macbeth 패턴에 대한 정면↔측면 Color Shift Δu`v` = sqrt((u`_60 - u`_0)^2 + (v`_60 - v`_0)^2)
        
        return:
        { 'Red': val, 'Green': val, ..., 'Western': val }  # 12개
        """
        macbeth_patterns = [
            "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow",
            "White", "Gray", "Darkskin", "Lightskin", "Asian", "Western"
        ] # 12개
        parameters_0 = [f"VAC_ColorShift_0_{p}" for p in macbeth_patterns]
        parameters_60 = [f"VAC_ColorShift_60_{p}" for p in macbeth_patterns]

        df_0 = self._load_measure_data(self.pk, components=('Cx', 'Cy'), 
                                       parameters=parameters_0, normalize_lv_flag=False)
        df_60 = self._load_measure_data(self.pk, components=('Cx', 'Cy'), 
                                        parameters=parameters_60, normalize_lv_flag=False)
        
        # logging.debug(f"0도 데이터:\n{df_0}")
        # logging.debug(f"60도 데이터:\n{df_60}")

        y2 = {}
        for mth_ptn in macbeth_patterns:
            try:
                cx_0 = df_0[(df_0['Pattern_Window'] == mth_ptn) & (df_0['Component'] == 'Cx')]['Data'].values[0]
                cy_0 = df_0[(df_0['Pattern_Window'] == mth_ptn) & (df_0['Component'] == 'Cy')]['Data'].values[0]
                cx_60 = df_60[(df_60['Pattern_Window'] == mth_ptn) & (df_60['Component'] == 'Cx')]['Data'].values[0]
                cy_60 = df_60[(df_60['Pattern_Window'] == mth_ptn) & (df_60['Component'] == 'Cy')]['Data'].values[0]

                denom_0 = (-2 * cx_0 + 12 * cy_0 + 3)
                denom_60 = (-2 * cx_60 + 12 * cy_60 + 3)
                u_0, v_0 = 4 * cx_0 / denom_0, 9 * cy_0 / denom_0
                u_60, v_60 = 4 * cx_60/denom_60, 9 * cy_60/denom_60

                delta_uv = np.sqrt((u_60 - u_0)**2 + (v_60 - v_0)**2)
                
                y2[mth_ptn] = float(delta_uv)
            except Exception:
                y2[mth_ptn] = 0.0
                
        return y2

    def prepare_Y(self, add_y0: bool = True, add_y1: bool = True, add_y2: bool = True,
                        y0_patterns=('W',), y1_patterns=('W',)):
        """
        병합된 최종 Y 딕셔너리 반환:
        {
          "Y0": { 'W': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)}, 
                  'R': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)},
                  'G': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)},
                  'B': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)},
                }
          "Y1": { 'W': (18,),
                  'R': (18,),
                  'G': (18,),
                  'B': (18,) 
                },
          "Y2": { 'Red': val, 
                  ..., 
                  'Western': val 
                }
        }
        """
        out = {}
        if add_y0:
            out["Y0"] = self.compute_Y0_struct(patterns=y0_patterns)
            
        if add_y1:
            out["Y1"] = self.compute_Y1_struct(patterns=y1_patterns)
            
        if add_y2:
            out["Y2"] = self.compute_Y2_struct()
        
        return out
    
    def debug_Y0_dataset(
        self,
        patterns=('W',),
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
            gray_samples = [0, 1, 32, 64, 128, 192, 254, 255]

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
        y0_patterns=('W',),
        y1_patterns=('W',),
        gray_samples_Y0=None,
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
        self.debug_Y2_dataset()
        
    def export_measure_data_to_csv(
        self,
        pk_list=None,
        parameters=None,
        components=('Lv', 'Cx', 'Cy'),
        normalize_lv_flag: bool = True,
        with_chart: bool = False,
        open_after: bool = True,
    ):
        """
        _load_measure_data()를 이용해 측정 데이터를 내보내는 통합 헬퍼.

        사용 모드 2가지:

        1) with_chart=False  (기본값)
           - pk_list 각각에 대해 _load_measure_data 결과를 CSV(1파일/PK)로 저장
           - 디버그용 원본 테이블 보기용

        2) with_chart=True
           - pk_list (여러 개 가능)에 대해,
             'VAC_Gamma_W_Gray____' + 'VAC_GammaLinearity_60_W_Gray____' 데이터를
             한 개의 Excel(xlsx)에서 PK별 시트로 저장하고,
             각 시트에 Nor.Lv(측면) 라인 차트까지 포함.
           - 이 모드는 내부에서 parameters/components를 자동으로 설정합니다.
        """
        # pk_list 정규화
        if pk_list is None:
            pk_list = [self.pk]
        elif isinstance(pk_list, int):
            pk_list = [pk_list]
        else:
            pk_list = list(pk_list)

        # -------------------------
        # ❶ CSV only 모드 (기본)
        # -------------------------
        if not with_chart:
            if parameters is None:
                raise ValueError(
                    "[export_measure_data_to_csv] with_chart=False 모드에서는 "
                    "'parameters' 인자가 필요합니다.\n"
                    "예: parameters=['VAC_Gamma_W_Gray____', 'VAC_GammaLinearity_60_W_Gray____']"
                )

            for pk in pk_list:
                df = self._load_measure_data(
                    pk=pk,
                    parameters=parameters,
                    components=components,
                    normalize_lv_flag=normalize_lv_flag,
                )

                if df is None or df.empty:
                    print(f"[export_measure_data_to_csv] PK={pk} 에 대해 로드된 데이터가 없습니다.")
                    continue

                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"_VAC_measure_pk{pk}.csv"
                )
                tmp_path = tmp.name
                tmp.close()

                df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
                print(f"[OK] measure data CSV saved → {tmp_path}")

                if open_after:
                    webbrowser.open(f"file://{os.path.abspath(tmp_path)}")

            return  # CSV 모드 종료

        # -------------------------
        # ❷ Excel + chart 모드
        #    (옛 load_multiple_pk_data_with_chart 통합)
        # -------------------------
        # 이 모드는 W 패턴 정면/측면 Gamma 데이터 전용으로 설계
        #   - 정면:  VAC_Gamma_W_Gray____
        #   - 측면:  VAC_GammaLinearity_60_W_Gray____
        front_param = "VAC_Gamma_W_Gray____"
        side_param  = "VAC_GammaLinearity_60_W_Gray____"

        # 임시 엑셀 파일 생성
        tmp_xlsx = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        xlsx_path = tmp_xlsx.name
        tmp_xlsx.close()

        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            for pk in pk_list:
                # ---- 정면 W Gamma 측정 데이터 (정규화 Lv 포함) ----
                df_front = self._load_measure_data(
                    pk=pk,
                    parameters=[front_param],
                    components=('Lv', 'Cx', 'Cy'),
                    normalize_lv_flag=True,  # Nor.Lv로 쓰기 위해 True
                )

                # ---- 측면 60도 W GammaLinearity 데이터 (Nor.Lv) ----
                df_side = self._load_measure_data(
                    pk=pk,
                    parameters=[side_param],
                    components=('Lv',),
                    normalize_lv_flag=True,  # 여기서도 Nor.Lv로 사용
                )

                sheet_name = f"PK_{pk}"

                # --------- 시트에 front/side 테이블 쓰기 ---------
                start_row_side = 0

                if df_front is not None and not df_front.empty:
                    # 정면: Pattern_Window='W' 인 것만 pivot
                    sub_f = df_front[
                        (df_front["Pattern_Window"] == "W")
                        & df_front["Component"].isin(["Lv", "Cx", "Cy"])
                    ].copy()

                    if not sub_f.empty:
                        front_pivot = (
                            sub_f
                            .pivot(index="Gray_Level", columns="Component", values="Data")
                            .reset_index()
                            .rename(columns={"Gray_Level": "Gray"})
                        )
                        front_pivot.to_excel(
                            writer, sheet_name=sheet_name,
                            startrow=0, index=False
                        )
                        start_row_side = len(front_pivot) + 3  # 아래에 side 테이블 배치

                if df_side is not None and not df_side.empty:
                    # 측면: Pattern_Window='W', Component='Lv' 만 사용 (이미 Nor.Lv)
                    sub_s = df_side[
                        (df_side["Pattern_Window"] == "W")
                        & (df_side["Component"] == "Lv")
                    ].copy()

                    if not sub_s.empty:
                        sub_s = sub_s.sort_values("Gray_Level")
                        side_df = sub_s[["Gray_Level", "Data"]].rename(
                            columns={"Gray_Level": "Gray", "Data": "Nor. Lv"}
                        )
                        side_df.to_excel(
                            writer, sheet_name=sheet_name,
                            startrow=start_row_side, index=False
                        )

                        # --------- Nor. Lv 차트 추가 ---------
                        worksheet = writer.sheets[sheet_name]
                        chart = workbook.add_chart({"type": "line"})

                        # 엑셀 내에서의 컬럼 위치 계산
                        col_gray = side_df.columns.get_loc("Gray")      # 보통 0
                        col_nlv  = side_df.columns.get_loc("Nor. Lv")   # 보통 1

                        first_row = start_row_side + 1     # 헤더 아래부터
                        last_row  = start_row_side + len(side_df)

                        chart.add_series({
                            "name": "Nor. Lv (60deg, W)",
                            "categories": [
                                sheet_name, first_row, col_gray,
                                last_row,  col_gray
                            ],
                            "values": [
                                sheet_name, first_row, col_nlv,
                                last_row,  col_nlv
                            ],
                        })
                        chart.set_title({"name": "Side View Nor. Lv (W, 60deg)"})
                        chart.set_x_axis({"name": "Gray"})
                        chart.set_y_axis({"name": "Nor. Lv"})

                        worksheet.insert_chart(0, 10, chart)

        print(f"[OK] Excel with charts saved → {xlsx_path}")

        if open_after:
            webbrowser.open(f"file://{os.path.abspath(xlsx_path)}")
            
if __name__ == "__main__":
    TARGET_PK = 3157
    BASE_PK = 3008
    BYPASS_PK = 3007
        
    builder = VACOutputBuilder(pk=TARGET_PK, ref_pk=BYPASS_PK)

    builder.export_measure_data_to_csv(
        pk_list = range(3008,3142),
        parameters = [
            "VAC_Gamma_W_Gray____",
            "VAC_GammaLinearity_60_W_Gray____",
        ],
        components=('Lv', 'Cx', 'Cy'),
        normalize_lv_flag=True,
        with_chart=True,
    )    
    
    # builder.debug_full_Y_dataset(
    #     y0_patterns=('W',),
    #     y1_patterns=('W',),
    # )
    



여기서 builder = VACOutputBuilder(pk=BASE_PK, ref_pk=BYPASS_PK)로 dGamma 출력하게 하려면 어떻게 해요?
