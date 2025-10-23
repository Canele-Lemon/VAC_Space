# ===== [ADD] 정규화 다운샘플 & 업샘플 =====
def _down4096_to_256_norm(self, arr4096):
    """4096 → 256 다운샘플 + [0,1] 정규화 (학습 스케일과 일치)"""
    a = np.asarray(arr4096, dtype=np.float32)
    idx = np.round(np.linspace(0, 4095, 256)).astype(int)
    return (a[idx] / 4095.0).astype(np.float32)

def _up256_to_4096_norm(self, arr256_norm):
    """[0,1] 256 → [0,1] 4096 업샘플 (TV 적용 전 마지막에만 12bit 변환)"""
    arr256_norm = np.asarray(arr256_norm, dtype=np.float32)
    x_small = np.linspace(0, 1, 256)
    x_big   = np.linspace(0, 1, 4096)
    return np.interp(x_big, x_small, arr256_norm).astype(np.float32)

def _to_tv_12bit(self, arr4096_norm):
    """[0,1] 4096 → 12bit 정수"""
    a = np.asarray(arr4096_norm, np.float32)
    return np.clip(np.round(a * 4095.0), 0, 4095).astype(int)

# ===== [ADD] 패널 원핫 =====
def _panel_onehot(self, panel_text: str):
    # 학습 때 쓰던 순서와 동일해야 합니다.
    PANEL_MAKER_CATEGORIES = ['HKC(H2)', 'HKC(H5)', 'BOE', 'CSOT', 'INX']
    v = np.zeros(len(PANEL_MAKER_CATEGORIES), np.float32)
    try:
        i = PANEL_MAKER_CATEGORIES.index(panel_text)
        v[i] = 1.0
    except ValueError:
        # 미스매치면 전부 0 (학습과 계약 유지)
        pass
    return v

# ===== [ADD] per-gray(W) 한 행 피처 (길이=18) =====
def _build_runtime_feature_row_W(self, lut256_norm: dict, gray: int,
                                 panel_text: str, frame_rate: float, model_year_2digit: float):
    """
    스키마(18):
    [R_Low, R_High, G_Low, G_High, B_Low, B_High, panel_onehot(5), frame_rate, model_year(2-digit), gray_norm, W,R,G,B]
    """
    row = [
        float(lut256_norm['R_Low'][gray]),  float(lut256_norm['R_High'][gray]),
        float(lut256_norm['G_Low'][gray]),  float(lut256_norm['G_High'][gray]),
        float(lut256_norm['B_Low'][gray]),  float(lut256_norm['B_High'][gray]),
    ]
    row.extend(self._panel_onehot(panel_text).tolist())
    row.append(float(frame_rate))
    row.append(float(model_year_2digit))      # 반드시 두 자리(예: 25.0)
    row.append(gray / 255.0)                  # gray_norm
    # W 패턴 one-hot
    row.extend([1.0, 0.0, 0.0, 0.0])
    return np.asarray(row, dtype=np.float32)

# ===== [ADD] DB JSON → 런타임 X(256×18) 생성 =====
def _build_runtime_X_from_db_json(self, vac_data_json: str):
    vac_dict = json.loads(vac_data_json)

    # 4096→256 정규화 (학습 스케일과 동일)
    lut256_norm = {
        "R_Low":  self._down4096_to_256_norm(vac_dict["RchannelLow"]),
        "R_High": self._down4096_to_256_norm(vac_dict["RchannelHigh"]),
        "G_Low":  self._down4096_to_256_norm(vac_dict["GchannelLow"]),
        "G_High": self._down4096_to_256_norm(vac_dict["GchannelHigh"]),
        "B_Low":  self._down4096_to_256_norm(vac_dict["BchannelLow"]),
        "B_High": self._down4096_to_256_norm(vac_dict["BchannelHigh"]),
    }

    # UI 메타 (model_year는 두 자리로 강제)
    panel_text, frame_rate, model_year_full = self._get_ui_meta()
    model_year_2digit = float(int(model_year_full) % 100)

    X_rows = [
        self._build_runtime_feature_row_W(
            lut256_norm, g,
            panel_text=panel_text,
            frame_rate=frame_rate,
            model_year_2digit=model_year_2digit
        )
        for g in range(256)
    ]
    X = np.vstack(X_rows).astype(np.float32)
    ctx = {"panel_text": panel_text, "frame_rate": frame_rate, "model_year_2digit": model_year_2digit}
    return X, lut256_norm, ctx

# ===== [ADD] 런타임 X 디버그 로깅 =====
def _debug_log_runtime_X(self, X: np.ndarray, ctx: dict, tag="[RUNTIME X]"):
    # 기대: X.shape=(256,18)
    try:
        D = X.shape[1]
    except Exception:
        D = None
    logging.debug(f"{tag} shape={getattr(X,'shape',None)}, dim={D}")
    if X is None or X.shape != (256, 18):
        logging.warning(f"{tag} 스키마 불일치: 기대 (256,18), 실제 {getattr(X,'shape',None)}")

    # 컬럼 해석을 위해 인덱스 슬라이스
    idx = {
        "LUT": slice(0,6),
        "panel_onehot": slice(6,11),
        "fr": 11,
        "my": 12,
        "gray_norm": 13,
        "p_oh": slice(14,18),
    }

    # 패널 원핫 합/원핫성
    p_sum = X[:, idx["panel_onehot"]].sum(axis=1)
    uniq = np.unique(p_sum)
    logging.debug(f"{tag} panel_onehot sum unique: {uniq[:8]} (expect 0 or 1)")
    logging.debug(f"{tag} ctx: panel='{ctx.get('panel_text')}', fr={ctx.get('frame_rate')}, my(2digit)={ctx.get('model_year_2digit')}")

    # 샘플 행 (0, 128, -1) & tail10
    def _fmt_row(i):
        r = X[i]
        lut = ", ".join(f"{v:.4f}" for v in r[idx["LUT"]])
        tail = ", ".join(f"{v:.4f}" for v in r[-10:])
        return f"idx={i:3d} | LUT6=[{lut}] | tail10=[{tail}]"
    logging.debug(f"{tag} sample: {_fmt_row(0)}")
    logging.debug(f"{tag} sample: {_fmt_row(128)}")
    logging.debug(f"{tag} sample: {_fmt_row(255)}")

    # 마지막 10개 행의 tail & 회귀 타깃이 없으니 gray_norm만 체크
    for i in range(246, 256):
        r = X[i]
        tail10 = tuple(float(x) for x in r[-10:])
        logging.debug(f"{tag} last10 idx={i:3d} | gray_norm={r[idx['gray_norm']]:.4f} | tail10={tail10}")
        
def _get_ui_meta(self):
    panel_text = self.ui.vac_cmb_PanelMaker.currentText().strip()

    fr_text = self.ui.vac_cmb_FrameRate.currentText().strip()
    fr_num = 0.0
    m = re.search(r'(\d+(?:\.\d+)?)', fr_text)
    if m: fr_num = float(m.group(1))

    my_text = self.ui.vac_cmb_ModelYear.currentText().strip() if hasattr(self.ui, "vac_cmb_ModelYear") else ""
    model_year = 0.0
    m = re.search(r'(\d{2,4})', my_text)
    if m:
        yy = int(m.group(1))
        model_year = float(yy % 100)   # ← 두 자리 강제
    return panel_text, fr_num, model_year

def _apply_vac_from_db_and_measure_on(self):
    panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
    fr    = self.ui.vac_cmb_FrameRate.currentText().strip()
    vac_pk, vac_version, vac_data = self._fetch_vac_by_model(panel, fr)
    if vac_data is None:
        logging.error(f"{panel}+{fr} 조합으로 매칭되는 VAC Data가 없습니다 - 종료")
        return

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # [ADD] 런타임 X(256×18) 생성 & 스키마 디버그 로깅
    try:
        X_runtime, lut256_norm, ctx = self._build_runtime_X_from_db_json(vac_data)
        self._debug_log_runtime_X(X_runtime, ctx, tag="[RUNTIME X from DB+UI]")
    except Exception as e:
        logging.exception("[RUNTIME X] build/debug failed")
        # 여기서 실패하면 예측/최적화 전에 스키마 문제로 조기 중단하도록 권장
        return
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


