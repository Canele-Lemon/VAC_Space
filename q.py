    def _load_mapping_j(self, csv_path=None):
        """
        LUT_index_mapping.csv 로부터 i(0..255) → j(0..4095) 매핑을 읽어 반환
        - 기본 경로: prepare_input.py 상위 폴더의 'LUT_index_mapping.csv'
        - 컬럼명: '8bit gray', '12bit LUT index'
        """
        import os, pandas as pd, numpy as np
        if csv_path is None:
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LUT_index_mapping.csv'))

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

    def prepare_X_delta_raw_with_mapping(self, ref_vac_info_pk: int):
        """
        ref(=W_VAC_Info.PK=ref_vac_info_pk)와 target(=self.PK가 가리키는 VAC_Set의 VAC_Info_PK)의
        각 채널 LUT(4096)에서, i→j 매핑 지점의 raw 12bit Δ를 256포인트로 뽑아온다.
        정규화 없음.
        return:
        {
          'lut_delta_raw': { 'R_Low':(256,), 'R_High':..., 'G_Low':..., 'G_High':..., 'B_Low':..., 'B_High':... },
          'meta': { 'panel_maker':(n,), 'frame_rate':float, 'model_year':float },
          'mapping_j': (256,) int32
        }
        """
        import numpy as np, logging

        # meta & target LUT
        info = self._load_vac_set_info_row(self.PK)
        if info is None:
            raise RuntimeError(f"[prepare_X_delta_raw_with_mapping] No VAC_SET row for PK={self.PK}")
        vac_info_pk_target = info["vac_info_pk"]
        meta_dict          = info["meta"]

        lut4096_target = self._load_vacdata_lut4096(vac_info_pk_target)
        if lut4096_target is None:
            raise RuntimeError(f"[prepare_X_delta_raw_with_mapping] No VAC_Data for VAC_Info_PK={vac_info_pk_target}")

        # ref LUT
        lut4096_ref = self._load_vacdata_lut4096(ref_vac_info_pk)
        if lut4096_ref is None:
            raise RuntimeError(f"[prepare_X_delta_raw_with_mapping] No REF VAC_Data for VAC_Info_PK={ref_vac_info_pk}")

        # mapping j (256,)
        j = self._load_mapping_j()  # 기본 경로 ../LUT_index_mapping.csv

        chan_list = ['R_Low','R_High','G_Low','G_High','B_Low','B_High']
        lut_delta_raw = {}
        for ch in chan_list:
            tgt = np.asarray(lut4096_target[ch], dtype=np.float32)
            ref = np.asarray(lut4096_ref[ch],    dtype=np.float32)
            if tgt.shape[0] != 4096 or ref.shape[0] != 4096:
                raise ValueError(f"[{ch}] LUT length must be 4096.")
            # 각 i의 j에서 Δ = target[j]-ref[j]
            lut_delta_raw[ch] = (tgt[j] - ref[j]).astype(np.float32)

        pack = {
            "lut_delta_raw": lut_delta_raw,
            "meta": meta_dict,
            "mapping_j": j.astype(np.int32)
        }
        logging.debug("[prepare_X_delta_raw_with_mapping] done.")
        return pack