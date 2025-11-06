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