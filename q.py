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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.db_config import engine
from config.app_config import PANEL_MAKER_CATEGORIES

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
    
    def _load_vac_set_info_row(self, pk: int):
        """
        내부 유틸:
        - VAC_SET_INFO_TABLE에서 pk 행 하나 가져오고
        - panel_maker one-hot, frame_rate, model_year, vac_info_pk를 뽑는다.
        - prepare_X0() / prepare_X_delta() 공통 사용
        """
        n_panel = len(PANEL_MAKER_CATEGORIES[0])

        query_set = f"""
        SELECT *
        FROM `{self.VAC_SET_INFO_TABLE}`
        WHERE `PK` = {pk}
        """
        df_set = pd.read_sql(query_set, engine)

        if df_set.empty:
            logging.error(f"[VACInputBuilder] No data found for PK={pk}")
            return None

        # model_year 전처리 ('Y23' -> 23 등)
        df_set['Model_Year'] = df_set['Model_Year'].str.replace('Y', '').astype(int)

        # panel maker one-hot
        panel_maker_encoder = OneHotEncoder(
            categories=PANEL_MAKER_CATEGORIES,
            sparse_output=False,
            handle_unknown='ignore'
        )
        panel_maker_oh = panel_maker_encoder.fit_transform(df_set[['Panel_Maker']])
        panel_maker_oh = panel_maker_oh[0].astype(np.float32)

        row = df_set.iloc[0]
        vac_info_pk = row['VAC_Info_PK']
        frame_rate = float(row['Frame_Rate'])
        model_year = float(row['Model_Year'])

        meta_dict = {
            "panel_maker": panel_maker_oh,
            "frame_rate": frame_rate,
            "model_year": model_year
        }

        return {
            "vac_info_pk": vac_info_pk,
            "meta": meta_dict,
            "n_panel": n_panel
        }
        
    def _load_vacdata_lut4096(self, vac_set_info_pk: int):
        """
        내부 유틸:
        - W_VAC_SET_Info 테이블에서 주어진 세트 PK(vac_set_info_pk)의 VAC_Info_PK를 먼저 조회한 뒤
        - 그 VAC_Info_PK를 사용해서 W_VAC_Info 테이블에서 VAC_Data(JSON)를 읽어온다.
        - 4096포인트 LUT 배열(dict)을 반환.

        반환 예:
        {
            "R_Low":  np.array([...], float32)  # len 4096, 정규화 전(raw)
            "R_High": ...
            ...
        }
        """
        # 1) 세트 테이블에서 VAC_Info_PK 조회
        query_set = f"""
        SELECT `VAC_Info_PK`
        FROM `{self.VAC_SET_INFO_TABLE}`
        WHERE `PK` = {vac_set_info_pk}
        """
        df_set = pd.read_sql(query_set, engine)
        if df_set.empty:
            logging.warning(f"[VACInputBuilder] No VAC_SET_Info found for PK={vac_set_info_pk}")
            return None

        vac_info_pk = int(df_set.iloc[0]["VAC_Info_PK"])

        # 2) VAC_Info_PK로 W_VAC_Info에서 VAC_Data 조회
        query_vacdata = f"""
        SELECT `VAC_Data`
        FROM `{self.VAC_DATA_TABLE}`
        WHERE `PK` = {vac_info_pk}
        """
        df_vacdata = pd.read_sql(query_vacdata, engine)
        if df_vacdata.empty:
            logging.warning(f"[VACInputBuilder] No VAC_Data found for VAC_Info_PK={vac_info_pk}")
            return None

        vacdata_dict = json.loads(df_vacdata.iloc[0]['VAC_Data'])

        channels = ['R_Low', 'R_High', 'G_Low', 'G_High', 'B_Low', 'B_High']
        lut4096 = {}
        for ch in channels:
            key = ch.replace("_", "channel")  # "R_Low" -> "RchannelLow"
            arr4096 = np.array(vacdata_dict.get(key, [0]*4096), dtype=np.float32)
            lut4096[ch] = arr4096

        return lut4096

    def _lut4096_to_lut256_norm(self, lut4096_dict: dict):
        """
        내부 유틸:
        - 4096포인트 LUT dict -> (정규화 후) 256포인트 LUT dict
        정규화: /4095.0
        다운샘플: downsample_lut()
        반환 예:
        {
            "R_Low":  (256,) float32 in [0,1]
            "R_High": (256,) ...
            ...
        }
        """
        lut256 = {}
        for ch, arr4096 in lut4096_dict.items():
            arr_norm = (arr4096.astype(np.float32) / 4095.0)
            lut256[ch] = self.downsample_lut(arr_norm).astype(np.float32)
        return lut256
    
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
        info = self._load_vac_set_info_row(self.PK)
        if info is None:
            return _empty_return()

        meta_dict          = info["meta"]

        lut4096_target = self._load_vacdata_lut4096(self.PK)
        if lut4096_target is None:
            return _empty_return()

        lut256_target = self._lut4096_to_lut256_norm(lut4096_target)

        return {
            "lut": lut256_target,
            "meta": meta_dict
        }
        
    def prepare_X_delta(self):
        """
        [자코비안/보정용 데이터셋 생성용]
        target PK (=self.PK)와
        reference LUT (= self.VAC_DATA_TABLE에서 PK=1인 행의 VAC_Data)를 비교하여
        ΔLUT(target - ref)을 256포인트 정규화 기준으로 반환.

        meta는 target 패널의 meta 그대로 사용.
        구조는 prepare_X0()와 동일:
        {
            "lut": { "R_Low": (256,), ... },   # 각 값 = target - ref
            "meta": { "panel_maker":..., "frame_rate":..., "model_year":... }
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

        # 1) target 쪽 정보 (현재 PK)
        info_target = self._load_vac_set_info_row(self.PK)
        if info_target is None:
            return _empty_return()

        meta_dict          = info_target["meta"]

        lut4096_target = self._load_vacdata_lut4096(self.PK)
        if lut4096_target is None:
            return _empty_return()

        # 2) reference 쪽 정보
        #    reference LUT은 VAC_DATA_TABLE의 PK=1 고정이라고 하셨습니다.
        lut4096_ref = self._load_vacdata_lut4096(vac_set_info_pk=1)
        if lut4096_ref is None:
            logging.warning("[VACInputBuilder] No reference LUT found at VAC_Info.PK=1, returning zeros.")
            return _empty_return()

        # 3) 둘 다 256포인트 정규화 LUT로 변환
        lut256_target = self._lut4096_to_lut256_norm(lut4096_target)  # dict of (256,)
        lut256_ref    = self._lut4096_to_lut256_norm(lut4096_ref)

        # 4) ΔLUT 계산 (target - ref)
        delta_lut256 = {}
        for ch in ['R_Low','R_High','G_Low','G_High','B_Low','B_High']:
            tgt_arr = lut256_target.get(ch, np.zeros(256, np.float32))
            ref_arr = lut256_ref.get(ch,    np.zeros(256, np.float32))
            delta_lut256[ch] = (tgt_arr - ref_arr).astype(np.float32)

        # 5) 반환
        return {
            "lut": delta_lut256,
            "meta": meta_dict
        }
        
    def _load_lut_index_mapping(self, csv_path=None):
        """
        LUT_index_mapping.csv 로부터 i(0..255) → j(0..4095) 매핑을 읽어 반환
        - 기본 경로: prepare_input.py 상위 폴더의 'LUT_index_mapping.csv'
        - 컬럼명: '8bit gray', '12bit LUT index'
        """
        if csv_path is None:
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'LUT_index_mapping.csv'))

        df = pd.read_csv(csv_path)
        # 컬럼명 방어
        col_i = None
        for c in df.columns:
            if str(c).strip().lower().replace(" ", "") in ("8bitgray", "gray", "i"):
                col_i = c; break
        if col_i is None:
            col_i = "8bit gray"
        col_j = None
        for c in df.columns:
            if str(c).strip().lower().replace(" ", "") in ("12bitlutindex", "lutindex", "j"):
                col_j = c; break
        if col_j is None:
            col_j = "12bit LUT index"

        # 256개 정렬 확보
        if len(df) < 256:
            raise ValueError(f"[Mapping] rows<{len(df)}> insufficient. Need 256 rows.")
        df = df.sort_values(by=col_i)
        j = df[col_j].to_numpy().astype(np.int32)
        if j.shape[0] > 256:
            j = j[:256]
        if j.shape[0] != 256:
            raise ValueError(f"[Mapping] mapping length must be 256, got {j.shape[0]}")
        # 클립
        j = np.clip(j, 0, 4095).astype(np.int32)
        return j
        
    def prepare_X_delta_raw_with_mapping(self, ref_vac_set_pk: int):
        """
        ref_vac_set_pk: W_VAC_SET_Info 테이블에서 'ref' LUT가 들어있는 세트 PK (예: 2582)
        return:
            {
            "lut_delta_raw": { "R_Low": (256,), ... },   # raw 12bit (target - ref) @ mapped j
            "meta": { panel_maker one-hot, frame_rate, model_year },
            "mapping_j": (256,) np.int32
            }
        """
        # 1) 타겟 메타/타겟 LUT(4096)
        info = self._load_vac_set_info_row(self.PK)
        if info is None:
            raise RuntimeError(f"[X_delta_raw] No VAC_SET_Info for PK={self.PK}")

        meta_dict = info["meta"]

        # 현재 세트 PK(self.PK)의 LUT
        lut4096_target = self._load_vacdata_lut4096(self.PK)
        if lut4096_target is None:
            raise RuntimeError(f"[X_delta_raw] No VAC_Data for VAC_SET_Info.PK={self.PK}")

        # 2) ref LUT(4096) — ref 세트 PK로부터
        lut4096_ref = self._load_vacdata_lut4096(ref_vac_set_pk)
        if lut4096_ref is None:
            raise RuntimeError(f"[X_delta_raw] No REF VAC_Data for VAC_SET_Info.PK={ref_vac_set_pk}")

        # 3) i(0..255) → j(0..4095) 매핑 로드 (CSV)
        j_map = self._load_lut_index_mapping()  # (256,) int

        # 4) 매핑 지점에서 raw delta(target - ref) 계산 (정규화 없음)
        delta = {}
        for ch in ['R_Low','R_High','G_Low','G_High','B_Low','B_High']:
            tgt = np.asarray(lut4096_target[ch], dtype=np.float32)
            ref = np.asarray(lut4096_ref[ch],    dtype=np.float32)
            if tgt.shape[0] != 4096 or ref.shape[0] != 4096:
                raise ValueError(f"[X_delta_raw] channel {ch}: need 4096-length arrays.")
            delta[ch] = (tgt[j_map] - ref[j_map]).astype(np.float32)

        return {"lut_delta_raw": delta, "meta": meta_dict, "mapping_j": j_map}

    def debug_dump_delta_with_mapping(self, pk=None, ref_vac_info_pk: int = 1, verbose_lut: bool = False):
        if pk is not None:
            self.PK = int(pk)

        # ΔLUT + 메타 + 매핑 먼저 불러오기
        pack = self.prepare_X_delta_raw_with_mapping(ref_vac_info_pk=ref_vac_info_pk)
        delta = pack["lut_delta_raw"]; meta = pack["meta"]; j_map = pack["mapping_j"]

        print(f"\n[DEBUG] ΔLUT(raw, target−ref@VAC_Info_PK={ref_vac_info_pk}) @ mapped indices for VAC_SET_Info.PK={self.PK}")
        print("[META]")
        print(f"  panel_maker one-hot: {meta['panel_maker']}")
        print(f"  frame_rate         : {meta['frame_rate']}")
        print(f"  model_year         : {meta['model_year']}")
        print("\n[MAPPING] j[0..10] =", j_map[:11].tolist(), "...")

        # ---- 여기서부터 추가: 원본 4096 LUT도 같이 로딩 ----
        info_target = self._load_vac_set_info_row(self.PK)
        vac_info_pk_target = info_target["vac_info_pk"]

        lut4096_target = self._load_vacdata_lut4096(vac_info_pk_target)
        lut4096_ref    = self._load_vacdata_lut4096(ref_vac_info_pk)

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


if __name__ == "__main__":
    builder = VACInputBuilder(pk=2757)
    builder.debug_dump_delta_with_mapping(pk=2757, ref_vac_info_pk=2744)


PS D:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project> & C:/python310/python.exe "d:/00 업무/00 가상화기술/00 색시야각 보상 최적화/VAC algorithm/VAC_Optimization_Project/src/data_preparation/prepare_input.py"
Traceback (most recent call last):
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\prepare_input.py", line 381, in <module>
    builder.debug_dump_delta_with_mapping(pk=2757, ref_vac_info_pk=2744)
  File "d:\00 업무\00 가상화기술\00 색시야각 보상 최적화\VAC algorithm\VAC_Optimization_Project\src\data_preparation\prepare_input.py", line 342, in debug_dump_delta_with_mapping
    pack = self.prepare_X_delta_raw_with_mapping(ref_vac_info_pk=ref_vac_info_pk)
TypeError: VACInputBuilder.prepare_X_delta_raw_with_mapping() got an unexpected keyword argument 'ref_vac_info_pk'
에러발생해요
