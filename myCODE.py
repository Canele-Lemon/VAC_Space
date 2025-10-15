# (1) 결과 저장용 버퍼 초기화 (OFF / ON 구분)
    self._off_store = {
        'Lv': np.zeros(256, dtype=float),
        'Cx': np.zeros(256, dtype=float),
        'Cy': np.zeros(256, dtype=float),
        'Gamma': np.zeros(256, dtype=float)
    }
    self._on_store = {
        'Lv': np.zeros(256, dtype=float),
        'Cx': np.zeros(256, dtype=float),
        'Cy': np.zeros(256, dtype=float),
        'Gamma': np.zeros(256, dtype=float)
    }