def _load_lut_mapping_high(self):
    """
    실행 py 파일 폴더에 있는 LUT_index_mapping.csv 를 읽어
    각 gray별 High LUT index를 저장.
    
    CSV 예시 가정:
        gray,R_High,G_High,B_High
        0,0,0,0
        1,16,16,16
        ...
    """
    if hasattr(self, "_lut_map_high") and self._lut_map_high is not None:
        return  # 이미 로드됨

    csv_path = cf.get_normalized_path(__file__, '.', 'LUT_index_mapping.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"LUT_index_mapping.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 🔧 컬럼명은 실제 파일에 맞게 조정 필요
    self._lut_map_high = {
        "R": df["R_High"].to_numpy(dtype=np.int32),
        "G": df["G_High"].to_numpy(dtype=np.int32),
        "B": df["B_High"].to_numpy(dtype=np.int32),
    }
    logging.info(f"[LUT MAP] loaded {csv_path}, shape={df.shape}")