# prepare_input.py
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import tempfile
import webbrowser
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config.db_config import engine
from src.config.app_config import PANEL_MAKER_CATEGORIES

logging.basicConfig(level=logging.DEBUG)

class VACInputBuilder:
    def __init__(self, pk: int):
        self.PK = pk
        self.VAC_SET_INFO_TABLE = "W_VAC_SET_Info"
        self.VAC_DATA_TABLE = "W_VAC_Info"        

    def downsample_lut(self, lut_4096):
        """
        Downsample 4096-point LUT to 256-point LUT using uniform sampling.
        """
        indices_256 = np.round(np.linspace(0, 4095, 256)).astype(int)
        return np.array(lut_4096)[indices_256]

    def prepare_X0(self):
        """
        특정 PK에 대한 구조화된 X0 반환
        return:
        {
            "lut": {  # 256포인트 LUT (정규화됨)
                "R_Low":  (256,) np.float32,
                "R_High": (256,) np.float32,
                "G_Low":  (256,) np.float32,
                "G_High": (256,) np.float32,
                "B_Low":  (256,) np.float32,
                "B_High": (256,) np.float32
            },
            "meta": {  # 메타 데이터를 key와 함께 제공
                "panel_maker": (n_panel,) np.float32,   # one-hot
                "frame_rate":   float,
                "model_year":   float
            }
        }
        """
        n_panel = len(PANEL_MAKER_CATEGORIES[0])
        def _empty_return():
            return {
                "lut": {k: np.zeros(256, np.float32)
                        for k in ['R_Low','R_High','G_Low','G_High','B_Low','B_High']},
                "meta": {
                    "panel_maker": np.zeros(n_panel, np.float32),
                    "frame_rate": 0.0,
                    "model_year": 0.0
                }
            }
        # === self.PK에 대한 SET_INFO 조회 및 Model_Year, Panel_Maker 데이터 전처리 ===
        query_set = f"""
        SELECT * FROM `{self.VAC_SET_INFO_TABLE}`
        WHERE `PK` = {self.PK}
        """
        df_set = pd.read_sql(query_set, engine)            
        if df_set.empty:
            logging.error(f"[VACInputBuilder] No data found for PK={self.PK}")
            return _empty_return()
        # logging.debug(f"PK={self.PK} 에서의 VAC_SET_INFO DataFrame:\n{df_set}")
        
        # Model_Year 숫자 변환
        df_set['Model_Year'] = df_set['Model_Year'].str.replace('Y', '').astype(int)

        # One-Hot Encode panel maker
        panel_maker_encoder = OneHotEncoder(
            categories=PANEL_MAKER_CATEGORIES,
            sparse_output=False,
            handle_unknown='ignore'  # 목록에 없는 제조사 들어오면 0으로 처리
        )
        panel_maker = panel_maker_encoder.fit_transform(df_set[['Panel_Maker']])

        row = df_set.iloc[0]
        vac_info_pk = row['VAC_Info_PK']
        frame_rate = float(row['Frame_Rate'])
        model_year = float(row['Model_Year'])
        panel_maker = panel_maker[0].astype(np.float32)

        # --- VAC_Data 로드 ---
        query_vacdata = f"""
        SELECT `VAC_Data` FROM `{self.VAC_DATA_TABLE}`
        WHERE `PK` = {vac_info_pk}
        """
        df_vacdata = pd.read_sql(query_vacdata, engine)
        if df_vacdata.empty:
            logging.warning(f"[VACInputBuilder] No VAC_Data found for VAC_Info_PK={vac_info_pk}")
            return _empty_return()

        vacdata_dict = json.loads(df_vacdata.iloc[0]['VAC_Data'])

        channels = ['R_Low', 'R_High', 'G_Low', 'G_High', 'B_Low', 'B_High']
        
        lut = {}
        for ch in channels:
            key = ch.replace("_", "channel")
            lut_key = np.array(vacdata_dict.get(key, [0]*4096), dtype=np.float32)
            lut_key /= 4095.0
            lut[ch] = self.downsample_lut(lut_key).astype(np.float32)  # (256,)
            
        # # for Debugging    
        # lut_df = pd.DataFrame(lut)
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='', encoding='utf-8') as tmp_file:
        #     lut_df.to_csv(tmp_file.name, index=False)
        #     webbrowser.open(f"file://{tmp_file.name}")  # Open in default CSV viewer (e.g., Excel)

        return {
            "lut": lut,
            "meta": {
                "panel_maker": panel_maker,
                "frame_rate": frame_rate,
                "model_year": model_year
            }
        }

# prepare_output.py
from src.config.db_config import engine

import logging
import tempfile
import webbrowser
import torch

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class VACOutputBuilder:
    def __init__(self, pk: int, reference_pk: int = 203):
        self.TARGET_PK = pk
        self.REFERENCE_PK = reference_pk
        self.MEASUREMENT_DATA_TABLE = "W_VAC_SET_Measure"
        
    def load_set_info_pk_data(self, pk):
        """
        지정한 pk(VAC_SET_Info_PK)에 해당하는 모든 측정 데이터를 DataFrame으로 반환합니다.
        - 모든 Parameter, Component, Data 컬럼 포함
        - 디버깅, 검증, CSV 저장 등에 활용 가능
        """
        query = f"""
        SELECT *
        FROM `{self.MEASUREMENT_DATA_TABLE}`
        WHERE `VAC_SET_Info_PK` = {pk}
        """
        df = pd.read_sql(query, engine)
        logging.debug(f"[load_set_info_pk_data] Loaded {len(df)} rows for pk={pk}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='', encoding='utf-8') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            webbrowser.open(f"file://{tmp_file.name}")  # Open in default CSV viewer (e.g., Excel)

        return df

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
                lv0 = lv_0_fixed.get(pattern, 0.0)
                lv255 = lv_255.get(pattern, 1.0)
                denom = lv255 - lv0 if lv255 != lv0 else 1.0
                sub['Data'] = (sub['Data'] - lv0) / denom
                normalized_lv.append(sub)

            lv_df = pd.concat(normalized_lv, ignore_index=True)
            # logging.debug(f"[STEP 3] Lv 정규화 후 lv_df:\n{lv_df.head()}")

        df = df[df['Component'] != 'Lv'].copy() # 기존 df에서 Lv 데이터를 제거
        
        if 'lv_df' in locals():
            df = pd.concat([df, lv_df], ignore_index=True)

        # df = pd.concat([df, lv_df], ignore_index=True) # 정규화된 Lv 데이터를 병합
        
        # logging.debug(f"[STEP 3] Lv 정규화 후 df:\n{df.head(6)}")
        
        return df
    
    def flatten_Y0(self, merged_df):
        """
        Flatten Y[0] merged DataFrame into 1D vector in the order:
        [W_0_Lv, W_0_Cx, W_0_Cy, W_1_Lv, ..., B_255_Cy]
        """
        pattern_order = ['W', 'R', 'G', 'B']
        component_order = ['Lv', 'Cx', 'Cy']
        gray_order = list(range(256))

        flat_list = []

        for pattern in pattern_order:
            for gray in gray_order:
                for comp in component_order:
                    row = merged_df[
                        (merged_df['Pattern_Window'] == pattern) &
                        (merged_df['Gray_Level'] == gray) &
                        (merged_df['Component'] == comp)
                    ]
                    if not row.empty:
                        flat_list.append(row.iloc[0]['Diff'])  # 방향성 유지
                    else:
                        flat_list.append(0.0)  # 누락된 경우 0으로 채움

        return np.array(flat_list, dtype=np.float32)

    def compute_Y0_struct(self):
        """
        Y[0] detailed: 패턴별(W/R/G/B) 정면 Gamma 특성 차이 w/ self.REFERENCE_PK (dGamma, dCx, dCy)
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
        } dGamma를 계산할 수 없는 경우 0으로 처리
        """
        parameters = [
            "VAC_Gamma_W_Gray____",
            "VAC_Gamma_R_Gray____",
            "VAC_Gamma_G_Gray____",
            "VAC_Gamma_B_Gray____"
        ]
        components = ('Lv', 'Cx', 'Cy')
        patterns = ('W', 'R', 'G', 'B')
        L = 256

        df_target = self._load_measure_data(self.TARGET_PK, parameters=parameters, components=components)
        df_ref = self._load_measure_data(self.REFERENCE_PK, parameters=parameters, components=components)

        if df_target.empty or df_ref.empty:
            logging.warning(f"[Y0] Missing data (PK={self.TARGET_PK}, Ref={self.REFERENCE_PK})")
            return {p: {k: np.zeros(L, np.float32) for k in ('dGamma','dCx','dCy')} for p in patterns}  # fallback: zero 구조

        def calc_gamma_array(df_lv_pattern: pd.DataFrame) -> np.ndarray:
            """
            nor.Lv = 0 또는 gray=0/255이면 NaN으로 남김
            """
            gamma = np.full(L, np.nan, dtype=np.float32)
            if not df_lv_pattern.empty:
                lv_dict = dict(zip(df_lv_pattern['Gray_Level'].to_numpy(),
                                df_lv_pattern['Data'].to_numpy(dtype=np.float32)))
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
            dGamma = gamma_t - gamma_r
            dGamma  = np.nan_to_num(dGamma, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            def diff_component(comp: str) -> np.ndarray:
                arr = np.zeros(L, np.float32)
                sub_t = df_target[(df_target['Pattern_Window'] == ptn) & (df_target['Component'] == comp)]
                sub_r = df_ref[(df_ref['Pattern_Window'] == ptn)      & (df_ref['Component'] == comp)]
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

            dCx = diff_component('Cx')
            dCy = diff_component('Cy')

            y0[ptn] = {
                'dGamma': dGamma.astype(np.float32),
                'dCx': dCx.astype(np.float32),
                'dCy': dCy.astype(np.float32)
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
        df = self._load_measure_data(self.TARGET_PK, parameters=parameters, components=('Lv','Cx','Cy'))
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
        df = self._load_measure_data(self.TARGET_PK, components=('Lv',), parameters=parameters)
        
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

        df_0 = self._load_measure_data(self.TARGET_PK, components=('Cx', 'Cy'), 
                                       parameters=parameters_0, normalize_lv_flag=False)
        df_60 = self._load_measure_data(self.TARGET_PK, components=('Cx', 'Cy'), 
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

    def prepare_Y(self, y1_patterns=('W',)):
        """
        최종 Y 딕셔너리 병합 반환:
        {
          "Y0": { 'W': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)}, 
                  'R': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)},
                  'G': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)},
                  'B': {'Gamma':(256,), 'Cx':(256,), 'Cy':(256,)},
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
        y0 = self.compute_Y0_struct_abs()
        y1 = self.compute_Y1_struct(patterns=y1_patterns)
        y2 = self.compute_Y2_struct()
        
        return {"Y0": y0, "Y1": y1, "Y2": y2}
        
# VAC_dataset.py
import sys
import torch
import os
import json
import pandas as pd
import numpy as np
import tempfile, webbrowser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from src.prepare_input import VACInputBuilder
from src.prepare_output import VACOutputBuilder

# ---------------------------------------------------------
# 상수/유틸
# ---------------------------------------------------------
_PATTERN_LIST = ['W', 'R', 'G', 'B']
_MACBETH_LIST = [
    "Red","Green","Blue","Cyan","Magenta","Yellow",
    "White","Gray","Darkskin","Lightskin","Asian","Western"
]
def _onehot(idx: int, size: int) -> np.ndarray:
    """
    범주형 인덱스를 원-핫 벡터로 변환.

    Parameters
    ----------
    idx : int
        활성화할 인덱스 (0 ~ size-1), 범위를 벗어나면 전체 0
    size : int
        벡터 길이

    Returns
    -------
    np.ndarray, shape (size,)
    """
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v

class VACDataset(Dataset):
    """
    VACDataset: VACInputBuilder와 VACOutputBuilder로부터
    PK 리스트 기반의 구조화된 X/Y를 수집하고
    모델 유형별로 학습용 (X, y) 행렬/벡터를 조립하는 클래스
    """

    def __init__(self, pk_list):
        """
        초기화
        :param pk_list: 사용할 PK 번호 리스트
        """
        self.pk_list = list(pk_list)
        self.samples = []   # 원본 구조 보관
        self._collect()

    # -----------------------------------------------------
    # 원본(X/Y) 수집: dict 그대로 보관
    # -----------------------------------------------------
    def _collect(self):
        for pk in self.pk_list:
            x_builder = VACInputBuilder(pk)
            y_builder = VACOutputBuilder(pk)
            X = x_builder.prepare_X0()   # {"lut": {...}, "meta": {...}}
            Y = y_builder.prepare_Y(y1_patterns=('W',))    # {"Y0": {...}, "Y1": {...}, "Y2": {...}}
            self.samples.append({"pk": pk, "X": X, "Y": Y})

    # -----------------------------------------------------
    # 내부: 특정 gray에서 피처 벡터 만들기
    # -----------------------------------------------------
    def _build_features_for_gray(self, X_dict, gray: int, add_pattern: str | None = None,
                                 channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High')) -> np.ndarray:
        """
        특정 gray(0~255)에서 6채널 LUT 값 + 메타 + gray_norm + (옵션)패턴 원핫을 붙여 피처 구성.

        feature = [
            R_Low[g], R_High[g], G_Low[g], G_High[g], B_Low[g], B_High[g],
            panel_onehot..., frame_rate, model_year,
            gray/255,
            (opt) pattern_onehot(4)
        ]
        """
        lut = X_dict["lut"]
        meta = X_dict["meta"]

        row = []
        for ch in channels:
            row.append(float(lut[ch][gray]))

        # 메타
        panel_vec = np.asarray(meta["panel_maker"], dtype=np.float32).tolist()
        row.extend(panel_vec)
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))

        # 그레이 정규화 인덱스
        row.append(gray / 255.0)

        # (옵션) 패턴 원핫
        if add_pattern is not None:
            pat_idx = _PATTERN_LIST.index(add_pattern) if add_pattern in _PATTERN_LIST else -1
            row.extend(_onehot(pat_idx, len(_PATTERN_LIST)).tolist())

        return np.asarray(row, dtype=np.float32)
    
    def _build_features_for_segment(self, X_dict, g: int, add_pattern: str | None = None,
                                    low_only: bool = False) -> np.ndarray:
        """
        구간(g -> g+1) 예측용 피처:
        [ LUT@g , LUT@(g+1) , (LUT@diff=g+1 - g) , meta(centered_gray_norm)/패턴 ]
        - low_only=True면 LUT 채널 중 R/G/B Low만 사용 (R_Low, G_Low, B_Low)
        - gray_norm은 세그먼트 중심값 (g+0.5)/255 로 교체하여 사용
        """
        # 안전 범위
        if not (0 <= g < 255):
            raise ValueError(f"segment g must be in [0, 254], got g={g}")

        # g, g+1 시점 피처 (한 행씩)
        f_g  = self._build_features_for_gray(X_dict, g,   add_pattern=add_pattern)
        f_g1 = self._build_features_for_gray(X_dict, g+1, add_pattern=add_pattern)

        # --- 1) LUT 부분 분리 및 (옵션) Low-only ---
        # 앞 6개가 LUT: [R_Low, R_High, G_Low, G_High, B_Low, B_High]
        lut_g  = f_g[:6].copy()
        lut_g1 = f_g1[:6].copy()
        if low_only:
            lut_idx = [0, 2, 4]  # Low만
            lut_g   = lut_g[lut_idx]
            lut_g1  = lut_g1[lut_idx]

        lut_diff = lut_g1 - lut_g  # 구간 차분

        # --- 2) meta tail 구성 (panel_onehot, frame_rate, model_year, gray_norm, pattern_onehot)
        # f_g 기준 tail 사용 (pattern/onehot 포함)
        meta_tail = f_g[6:].copy()

        # --- 3) gray_norm을 세그먼트 중심값으로 교체 ---
        # panel_onehot 길이(K)는 meta에서 확인
        K = len(X_dict["meta"]["panel_maker"])  # one-hot 길이
        # tail 배열 내에서 gray_norm의 위치:
        # tail = [panel_onehot(K), frame_rate(1), model_year(1), gray_norm(1), pattern_onehot(4)]
        idx_gray_in_tail = K + 2
        if idx_gray_in_tail >= meta_tail.shape[0]:
            # 방어: 스키마가 바뀐 경우를 대비
            raise IndexError("gray_norm index is out of range in meta_tail; "
                            "check _build_features_for_gray feature layout.")
        # 세그먼트 중심값
        meta_tail[idx_gray_in_tail] = (g + 0.5) / 255.0

        # --- 4) 최종 피처 결합 ---
        feat = np.concatenate([lut_g, lut_g1, lut_diff, meta_tail]).astype(np.float32)
        return feat

    # -----------------------------------------------------
    # 1) 멀티타깃 일괄학습용 (전체 플랫)
    #    X: LUT(6*256) + meta
    #    Y: Y0(4*3*256) + Y1(4*255) + Y2(12)
    # -----------------------------------------------------
    def build_multitarget_flat(self, include=('Y0', 'Y1', 'Y2')):
        """
        멀티타깃 일괄학습(벡터 플랫) 데이터셋 생성.

        Returns
        -------
        X_mat : np.ndarray, shape (N, Dx)
            N = 샘플 수 (PK 개수),
            Dx = 6*256 + |panel_onehot| + 2(frame_rate, model_year)
        Y_mat : np.ndarray, shape (N, Dy)
            Dy = sum(include):
                - Y0: 4패턴 * 3컴포넌트 * 256 = 3072
                - Y1: 'W' 패턴 * 255 = 255
                - Y2: 12
              예) include=('Y0','Y1','Y2') → 3339
        """
        X_rows, Y_rows = [], []
        for s in self.samples:
            Xd = s["X"]; Yd = s["Y"]
            # X 플랫
            lut = Xd["lut"]; meta = Xd["meta"]
            x_flat = np.concatenate([
                lut['R_Low'], lut['R_High'],
                lut['G_Low'], lut['G_High'],
                lut['B_Low'], lut['B_High'],
                meta['panel_maker'].astype(np.float32),
                np.array([meta['frame_rate'], meta['model_year']], dtype=np.float32)
            ]).astype(np.float32)

            # Y 플랫
            y_parts = []
            if 'Y0' in include:
                for p in _PATTERN_LIST:
                    y_parts.append(np.nan_to_num(Yd['Y0'][p]['Gamma'], nan=0.0, posinf=0.0, neginf=0.0))
                    y_parts.append(np.nan_to_num(Yd['Y0'][p]['Cx'],    nan=0.0, posinf=0.0, neginf=0.0))
                    y_parts.append(np.nan_to_num(Yd['Y0'][p]['Cy'],    nan=0.0, posinf=0.0, neginf=0.0))
            if 'Y1' in include:
                y_parts.append(Yd['Y1']['W'])
            if 'Y2' in include:
                y_parts.append(np.array([Yd['Y2'][m] for m in _MACBETH_LIST], dtype=np.float32))

            y_flat = np.concatenate(y_parts).astype(np.float32)
            X_rows.append(x_flat)
            Y_rows.append(y_flat)

        X_mat = np.vstack(X_rows).astype(np.float32)
        Y_mat = np.vstack(Y_rows).astype(np.float32)
        return X_mat, Y_mat

    # -----------------------------------------------------
    # 2) Y0(계조별 dGamma/dCx/dCy) 회귀 (선형 추세 등)
    #    행 단위: (pk, pattern, gray)
    # -----------------------------------------------------
    # def build_per_gray_y0(self, component='Gamma', patterns=('W','R','G','B')):
    #     """
    #     Y0(계조별 Gamma/Cx/Cy) 단일 스칼라 회귀용 데이터셋.

    #     Parameters
    #     ----------
    #     component : {'Gamma','Cx','Cy'}
    #     patterns : tuple[str]

    #     Returns
    #     -------
    #     X_mat : np.ndarray, shape (N * len(patterns) * 256, Dx)
    #         행 단위 = (pk, pattern, gray)
    #     y_vec : np.ndarray, shape (N * len(patterns) * 256,)
    #         타깃 스칼라 (선택한 component 값)
    #     """
        # assert component in ('Gamma', 'Cx', 'Cy')
        # X_rows, y_vals = [], []
        # for s in self.samples:
        #     Xd = s["X"]; Yd = s["Y"]
        #     for p in patterns:
        #         y_vec = Yd['Y0'][p][component]  # (256,)
        #         for g in range(256):
        #             y_val = y_vec[g]
        #             if not np.isfinite(y_val):     # NaN/inf는 스킵
        #                 continue
        #             X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
        #             y_vals.append(float(y_val))
        # X_mat = np.vstack(X_rows).astype(np.float32)
        # y_vec = np.asarray(y_vals, dtype=np.float32)
        # return X_mat, y_vec
    def build_per_gray_y0(self, component='Gamma', patterns=('W','R','G','B')):
        assert component in ('Gamma', 'Cx', 'Cy')
        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk = s["pk"]
            Xd = s["X"]; Yd = s["Y"]
            for p in patterns:
                y_vec = Yd['Y0'][p][component]  # (256,)
                for g in range(256):
                    y_val = y_vec[g]
                    if not np.isfinite(y_val):   # NaN/inf는 스킵
                        continue
                    X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
                    y_vals.append(float(y_val))
                    groups.append(pk)            # ← 유지된 행에 대해 pk를 같이 쌓기

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

    # -----------------------------------------------------
    # 3) Y1(측면 slope) 회귀 (gray 0~254)
    #    행 단위: (pk, pattern, gray_segment)
    # -----------------------------------------------------
    def build_per_gray_y1(self, patterns=('W',), use_segment_features=True, low_only=True):
        """
        Y1(측면 중계조 slope) 단일 스칼라 회귀용 데이터셋.

        Parameters
        ----------
        patterns : tuple[str]

        Returns
        -------
        X_mat : np.ndarray, shape (N * len(patterns) * 255, Dx)
            행 단위 = (pk, pattern, gray_segment)
        y_vec : np.ndarray, shape (N * len(patterns) * 255,)
            slope 값
        """
        X_rows, y_vals = [], []
        for s in self.samples:
            Xd = s["X"]; Yd = s["Y"]
            for p in patterns:
                slope = Yd['Y1'][p]  # (255,)
                for g in range(255):
                    if use_segment_features:
                        X_rows.append(self._build_features_for_segment(Xd, g, add_pattern=p, low_only=low_only))
                    else:
                        # g 시점만
                        X_rows.append(self._build_features_for_gray(Xd, g, add_pattern=p))
                    y_vals.append(float(slope[g]))
        X_mat = np.vstack(X_rows).astype(np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        return X_mat, y_vec

    # -----------------------------------------------------
    # 4) Y2(Δu'v') 회귀: Macbeth 12패치 스칼라
    #    행 단위: (pk, macbeth_patch)
    # -----------------------------------------------------
    def build_y2_macbeth(self, use_lut_summary: bool = True):
        """
        Y2(Δu'v') 스칼라 회귀용 데이터셋.

        Parameters
        ----------
        use_lut_summary : bool
            True이면 LUT 요약(채널별 mean, 9pt에서의 LUT값)을 포함

        Returns
        -------
        X_mat : np.ndarray, shape (N * 12, Dx)
            행 단위 = (pk, macbeth_patch)
        y_vec : np.ndarray, shape (N * 12,)
            Δu'v' 값
        """
        X_rows, y_vals = [], []
        # LUT 9포인트 인덱스 (4096 기준)
        lut_points = [0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4095]
        
        for s in self.samples:
            Xd = s["X"]
            Yd = s["Y"]
            meta = Xd["meta"]
            
            feats = []
            # --- (1) 메타 데이터 ---
            feats.extend(meta['panel_maker'].astype(np.float32).tolist())
            feats.append(float(meta['frame_rate']))
            feats.append(float(meta['model_year']))

            # --- (2) LUT summary ---
            if use_lut_summary:
                lut = Xd["lut"]
                # 간단 요약: 채널별 mean, 9 포인트 값
                for ch in ["R_Low", "R_High", "G_Low", "G_High", "B_Low", "B_High"]:
                    arr = np.asarray(lut[ch], dtype=np.float32)
                    arr = np.clip(arr, 0.0, 1.0)  # 안전 장치
                    
                    # mean
                    feats.append(float(arr.mean()))
                    # 9포인트 샘플 (256포인트 기준으로 리샘플되어 있음)
                    n = len(arr)
                    for p in lut_points:
                        # 4096기준 인덱스 → 256포인트 기준으로 보정
                        idx = int(round(p / 16))  # 4096→256 매핑
                        idx = min(max(idx, 0), n - 1)
                        feats.append(float(arr[idx]))

            feats = np.asarray(feats, dtype=np.float32)

            # --- (3) Macbeth 12패치 반복 ---
            for patch in _MACBETH_LIST:
                X_rows.append(feats)                    # 동일 메타/요약으로 12행 생성
                y_vals.append(float(Yd['Y2'][patch]))

        X_mat = np.vstack(X_rows).astype(np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        return X_mat, y_vec

    # -----------------------------------------------------
    # 5) 잔차 보정용: 1차 모델 예측 벡터로부터 y_true - y_pred 생성
    # -----------------------------------------------------
    def build_residual_dataset(self, builder_fn, base_pred, **builder_kwargs):
        """
        잔차 보정용 데이터셋 (예: 선형 회귀 → RF 잔차 보정)

        Parameters
        ----------
        builder_fn : callable
            예) self.build_per_gray_y0, self.build_per_gray_y1 등
        base_pred : array-like
            1차 모델 예측 벡터 (y_true와 동일한 순서/길이)
        builder_kwargs : dict
            builder_fn에 전달할 파라미터

        Returns
        -------
        X_mat : np.ndarray, shape (M, Dx)
        y_resid : np.ndarray, shape (M,)
            y_true - base_pred
        """
        X_mat, y_true = builder_fn(**builder_kwargs)
        base_pred = np.asarray(base_pred, dtype=np.float32).reshape(-1)
        if base_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f"base_pred length {base_pred.shape[0]} != y_true length {y_true.shape[0]}")
        y_resid = (y_true - base_pred).astype(np.float32)
        return X_mat, y_resid

    # -----------------------------------------------------
    # (선택) 파이토치 호환: 길이/인덱싱
    # -----------------------------------------------------
    def __len__(self):
        # 파이토치 DataLoader로 직접 쓸 계획이 있다면,
        # 원하는 빌더 출력으로 커스텀 Dataset을 따로 구성하는 것을 권장합니다.
        return len(self.samples)

    def __getitem__(self, idx):
        # 원본 dict를 그대로 반환 (torch 학습 시엔 별도 빌더로 만든 X,y를 쓰세요)
        return self.samples[idx]
