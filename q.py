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

    def _load_vacdata_lut4096(self, vac_info_pk: int):
        """
        내부 유틸:
        - VAC_DATA_TABLE에서 VAC_Data(JSON)를 읽어와서
          4096포인트 LUT 배열(dict)을 반환
        - 채널명 매핑은 prepare_X0()과 동일하게 맞춘다.
        반환 예:
        {
            "R_Low":  np.array([...], float32)  # len 4096, 정규화 전(raw)
            "R_High": ...
            ...
        }
        """
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
        기존과 동일:
        target PK (=self.PK)의 절대 LUT (정규화된 256포인트) + meta
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

        vac_info_pk_target = info["vac_info_pk"]
        meta_dict          = info["meta"]

        lut4096_target = self._load_vacdata_lut4096(vac_info_pk_target)
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

        vac_info_pk_target = info_target["vac_info_pk"]
        meta_dict          = info_target["meta"]

        lut4096_target = self._load_vacdata_lut4096(vac_info_pk_target)
        if lut4096_target is None:
            return _empty_return()

        # 2) reference 쪽 정보
        #    reference LUT은 VAC_DATA_TABLE의 PK=1 고정이라고 하셨습니다.
        lut4096_ref = self._load_vacdata_lut4096(vac_info_pk=1)
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