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
    def __init__(self, pk: int, ref_pk: int = 2154):
        self.pk = pk
        self.ref_pk = ref_pk
        
        self.MEASUREMENT_INFO_TABLE = "W_VAC_SET_Info"
        self.MEASUREMENT_DATA_TABLE = "W_VAC_SET_Measure"
        
    def load_set_info_pk_data(self, pk):
        """
        pk : `W_VAC_SET_Info` 테이블의 `PK`
        
        지정한 pk에 해당하는 측정 데이터를 DataFrame으로 반환
        
        return :
        - df_front_pivot: VAC_Gamma_W_Gray
        - df_side_pivot: VAC_GammaLinearity_60_W_Gray

        """
        query = f"""
        SELECT `VAC_SET_Info_PK`, `Parameter`, `Component`, `Data`
        FROM `{self.MEASUREMENT_DATA_TABLE}`
        WHERE `VAC_SET_Info_PK` = {pk}
        """
        df_all = pd.read_sql(query, engine)
        
        df_front = df_all[df_all['Parameter'].str.contains('VAC_Gamma_W_Gray', na=False)]
        df_front_pivot = df_front.pivot(index=['VAC_SET_Info_PK', 'Parameter'], columns='Component', values='Data').reset_index()
        df_front_pivot.insert(0, 'Gray', range(len(df_front_pivot)))
        lv_values = pd.to_numeric(df_front_pivot['Lv'], errors='coerce')
        lv_min, lv_max = lv_values.min(), lv_values.max()
        
        def calc_gamma(lv_on_g, g):
            if not np.isfinite(lv_on_g) or lv_max <= lv_min:
                return float("nan")
            nor = (lv_on_g - lv_min) / (lv_max - lv_min)
            gray_norm = g / 255.0
            if nor <= 0 or gray_norm <= 0 or gray_norm >= 1:
                return float("nan")
            return float(np.log(nor) / np.log(gray_norm))
        
        df_front_pivot['Gamma'] = [calc_gamma(lv, g) for lv, g in zip(lv_values, df_front_pivot['Gray'])]

        df_side = df_all[df_all['Parameter'].str.contains('VAC_GammaLinearity_60_W_Gray', na=False)]
        df_side_pivot = df_side.pivot(index=['VAC_SET_Info_PK', 'Parameter'], columns='Component', values='Data').reset_index()
        df_side_pivot.insert(0, 'Gray', range(len(df_side_pivot)))
        lv_side_values = pd.to_numeric(df_side_pivot['Lv'], errors='coerce')
        lv_side_min, lv_side_max = lv_side_values.min(), lv_side_values.max()
        df_side_pivot['Nor. Lv'] = (lv_side_values - lv_side_min) / (lv_side_max - lv_side_min)
  
        query_version = f"""
        SELECT `VAC_Version`
        FROM `{self.MEASUREMENT_INFO_TABLE}`
        WHERE `PK` = {pk}
        """
        version_df = pd.read_sql(query_version, engine)
        vac_version = version_df['VAC_Version'].iloc[0] if not version_df.empty else f"PK_{pk}"
        
        logging.debug(f"[load_set_info_pk_data] Loading complete for `VAC_SET_Info_PK`={pk} from table=`{self.MEASUREMENT_DATA_TABLE}` (VAC Version={vac_version})")
        
        return vac_version, df_front_pivot, df_side_pivot

    def load_multiple_pk_data_with_chart(self, pk_list):
        """
        pk 리스트를 받아서 하나의 Excel 파일에 pk별 시트 생성
        각 시트에:
        - VAC_Gamma_W_Gray 데이터
        - VAC_GammaLinearity_60_W_Gray 데이터
        - Gray vs Lv_normalized 차트
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine='xlsxwriter') as writer:
                workbook = writer.book

                for pk in pk_list:
                    vac_version, df_front_pivot, df_side_pivot = self.load_set_info_pk_data(pk)
                    sheet_name = str(vac_version)

                    # 데이터 쓰기
                    df_front_pivot.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
                    start_row_side = len(df_front_pivot) + 3
                    df_side_pivot.to_excel(writer, sheet_name=sheet_name, startrow=start_row_side, index=False)

                    # 차트 추가
                    worksheet = writer.sheets[sheet_name]
                    chart = workbook.add_chart({'type': 'line'})
                    chart.add_series({
                        'name': 'Nor. Lv',
                        'categories': [sheet_name, start_row_side + 1, 0, start_row_side + len(df_side_pivot), 0],  # Gray
                        'values': [sheet_name, start_row_side + 1, df_side_pivot.columns.get_loc('Nor. Lv'),
                                start_row_side + len(df_side_pivot), df_side_pivot.columns.get_loc('Nor. Lv')],
                    })
                    chart.set_title({'name': 'Side View Luminance'})
                    chart.set_x_axis({'name': 'Gray'})
                    chart.set_y_axis({'name': 'Nor. Lv'})
                    worksheet.insert_chart(0, 10, chart)

            webbrowser.open(f"file://{tmp_file.name}")

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
        # logging.debug(f"[STEP 1] Raw data loaded:\n{df.head()}")
        
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

    def compute_Y0_struct(self, patterns=('W', 'R', 'G', 'B')):
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
        Y[1] detailed: 측면 중계조 표현력 (Nor.Lv(gray+1) - Nor.Lv(gray)) / (1/255)
        return:
        *patterns=('W','R','G','B') 선택 시,
        { 'W': (255,), 'R': (255,), 'G': (255,), 'B': (255,) }
        """
        parameters = [
            "VAC_GammaLinearity_60_W_Gray____",
            "VAC_GammaLinearity_60_R_Gray____",
            "VAC_GammaLinearity_60_G_Gray____",
            "VAC_GammaLinearity_60_B_Gray____"
        ]
        df = self._load_measure_data(self.pk, components=('Lv',), parameters=parameters)
        
        L = 256
        y1 = {}
        for ptn in patterns:
            lv = df[(df['Pattern_Window'] == ptn) & (df['Component'] == 'Lv')].sort_values('Gray_Level')['Data'].to_numpy()
            if lv.size == L:
                slope = 255.0 * (lv[1:] - lv[:-1])  # (255,)
            else:
                slope = np.zeros(L-1, np.float32)
            y1[ptn] = slope.astype(np.float32)
            
        return y1

    def compute_Y2_struct(self):
        """
        Y[2] detailed: Macbeth 패턴에 대한 정면<->측면 Color Shift Δu`v` = sqrt((u`_60 - u`_0)^2 + (v`_60 - v`_0)^2)
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
                        y0_patterns=('W', 'R', 'G', 'B'), y1_patterns=('W',)):
        """
        최종 Y 딕셔너리 병합 반환:
        {
          "Y0": { 'W': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)}, 
                  'R': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)},
                  'G': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)},
                  'B': {'dGamma':(256,), 'dCx':(256,), 'dCy':(256,)},
                }
          "Y1": { 'W': (255,),
                  'R': (255,),
                  'G': (255,),
                  'B': (255,) 
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

if __name__ == "__main__":
    pk_list = list(range(2973, 2984))
    builder = VACOutputBuilder(pk=pk_list[0], ref_pk=pk_list[1])
    builder.load_multiple_pk_data_with_chart(pk_list)
