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

여기서 WHERE `PK` = {vac_info_pk} 부분에 vac_info_pk말고 다른것이 들어가야 하는거같아요. 

먼저 self.VAC_SET_INFO_TABLE = "W_VAC_SET_Info" 테이블에서, self.PK = pk행의 `VAC_Info_PK` 값을 조회한 다음에 이 값을
FROM `{self.VAC_DATA_TABLE}`테이블의 `PK`행에서의 `VAC_Data`의 longtext를 가져오는 방식이 맞아요.
