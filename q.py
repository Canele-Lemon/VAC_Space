def _extract_model_contract(self):
    """
    pkl 안에 학습시 저장해둔 메타(있다면)를 꺼내 피처 계약을 구성.
    - 기대 필드(있을 수도/없을 수도): 
      meta = {
        "panel_categories": ["HKC(H2)", "HKC(H5)", "BOE", "CSOT", "INX"],
        "pattern_order": ["W","R","G","B"],
        "feature_names": [...],            # 훈련 스크립트에서 저장했을 때
        "lut_scale": "0..1"                # LUT 정규화 기대 스케일
      }
    """
    # 기본 폴백 (훈련과 동일해야 함: 직접 학습 스크립트 확인!)
    default_panels  = ["HKC(H2)", "HKC(H5)", "BOE", "CSOT", "INX"]
    default_patterns= ["W","R","G","B"]

    # 아무 모델에서나 meta를 시도 추출
    any_model = next(iter(self.models_Y0_bundle.values()))
    meta = any_model.get("meta", {}) if isinstance(any_model, dict) else {}

    panels   = meta.get("panel_categories", default_panels)
    patterns = meta.get("pattern_order",   default_patterns)
    featnames= meta.get("feature_names",   None)
    lut_scale= meta.get("lut_scale",       "0..1")

    return {
        "panel_categories": panels,
        "pattern_order": patterns,
        "feature_names": featnames,   # 있으면 열 순서 검증에 쓰기
        "lut_scale": lut_scale
    }

def _build_feature_matrix_W_checked(self, lut256_dict, *, panel_text, frame_rate, model_year):
    """
    'W' 패턴 256행 피처 매트릭스를 계약에 맞춰 생성하고,
    - n_features 일치 검사
    - 스케일/범위 검사
    - one-hot 카테고리 순서 검사
    를 수행한 뒤 (X, contract) 반환
    """
    contract = self._extract_model_contract()
    panel_cats  = contract["panel_categories"]
    pattern_ord = contract["pattern_order"]

    # 1) LUT 0..1 정규화
    def _norm01(a): 
        return np.clip(np.asarray(a, np.float32)/4095.0, 0.0, 1.0)
    R_L = _norm01(lut256_dict["R_Low"])
    R_H = _norm01(lut256_dict["R_High"])
    G_L = _norm01(lut256_dict["G_Low"])
    G_H = _norm01(lut256_dict["G_High"])
    B_L = _norm01(lut256_dict["B_Low"])
    B_H = _norm01(lut256_dict["B_High"])

    # 2) panel one-hot (훈련 순서 고정)
    panel_oh = np.zeros(len(panel_cats), np.float32)
    if panel_text in panel_cats:
        panel_oh[panel_cats.index(panel_text)] = 1.0
    else:
        logging.warning(f"[Predict/Contract] panel '{panel_text}' not in training cats {panel_cats}. (all-zero one-hot)")

    # 3) pattern one-hot 순서 확인 (W 가 index=0이어야 우리의 가정과 일치)
    if pattern_ord[0] not in ("W","White","white"):
        logging.warning(f"[Predict/Contract] training pattern order starts with {pattern_ord[0]} — expected 'W'. This must match training!")
    patt_W = np.zeros(len(pattern_ord), np.float32)
    try:
        patt_W[pattern_ord.index("W")] = 1.0
    except ValueError:
        # 훈련에서 "White"로 저장했을 수도
        if "White" in pattern_ord:
            patt_W[pattern_ord.index("White")] = 1.0
        else:
            logging.warning(f"[Predict/Contract] 'W' or 'White' not found in training pattern_order={pattern_ord}.")
            # 어쩔 수 없이 첫 칸에 1
            patt_W[0] = 1.0

    # 4) 행 단위 생성
    gray = np.arange(256, dtype=np.float32)
    gray_norm = gray/255.0
    Kp = len(panel_oh)
    Kpat = len(patt_W)

    # 기대 피처 순서: [R_L,R_H,G_L,G_H,B_L,B_H] + panel_oh + frame_rate + model_year + gray_norm + patt_W
    X = np.zeros((256, 6 + Kp + 2 + 1 + Kpat), dtype=np.float32)
    X[:,0]=R_L; X[:,1]=R_H; X[:,2]=G_L; X[:,3]=G_H; X[:,4]=B_L; X[:,5]=B_H
    X[:,6:6+Kp] = panel_oh.reshape(1,-1)
    X[:,6+Kp]   = float(frame_rate)
    X[:,6+Kp+1] = float(model_year)
    X[:,6+Kp+2] = gray_norm
    X[:,6+Kp+3:6+Kp+3+Kpat] = patt_W.reshape(1,-1)

    # 5) n_features 검증 (각 모델과 동일해야 함)
    for comp in ("Gamma","Cx","Cy"):
        lm = self.models_Y0_bundle[comp]["linear_model"]
        exp = getattr(lm, "n_features_in_", None)
        if exp is None and hasattr(lm, "coef_"):
            exp = lm.coef_.shape[1]
        if exp is not None and X.shape[1] != exp:
            logging.error(f"[Predict/Contract] n_features mismatch for {comp}: X={X.shape[1]} vs model={exp}")
        # RF도 체크
        rf = self.models_Y0_bundle[comp]["rf_residual"]
        if hasattr(rf, "n_features_in_") and rf.n_features_in_ != X.shape[1]:
            logging.error(f"[Predict/Contract] RF n_features mismatch for {comp}: X={X.shape[1]} vs RF={rf.n_features_in_}")

    # 6) 스케일/범위 로그
    def _mm(a): 
        return float(np.nanmin(a)), float(np.nanmax(a))
    logging.debug(f"[Predict/Contract] LUT(0..1) min/max — R_L{_mm(R_L)}, R_H{_mm(R_H)}, G_L{_mm(G_L)}, G_H{_mm(G_H)}, B_L{_mm(B_L)}, B_H{_mm(B_H)}")
    logging.debug(f"[Predict/Contract] meta — fr={frame_rate}, model_year={model_year}, gray_norm[0]={gray_norm[0]},[-1]={gray_norm[-1]}")
    logging.debug(f"[Predict/Contract] panel one-hot={panel_oh.tolist()}, pattern one-hot(W)={patt_W.tolist()}")
    return X, contract

def debug_check_prediction_contract_once(self):
    """
    - DB LUT(또는 캐시 LUT)를 4096→256 다운샘플해 계약대로 X를 만들고
    - 각 모델의 n_features, 예측 결과 통계(평균/표준편차)를 로그로 확인
    - g=128 한 줄의 피처를 상세 출력
    """
    # 1) 현재 사용할 LUT 소스 확보 (DB 읽은 것 또는 예측 LUT)
    if hasattr(self, "_vac_dict_cache") and self._vac_dict_cache:
        src = self._vac_dict_cache
    elif hasattr(self, "_vac_dict_last_preview") and self._vac_dict_last_preview:
        src = self._vac_dict_last_preview
    else:
        logging.error("[Predict/Debug] No LUT source available (need _vac_dict_cache or _vac_dict_last_preview).")
        return

    lut256 = {
        "R_Low":  self._down4096_to_256(src["RchannelLow"]),
        "R_High": self._down4096_to_256(src["RchannelHigh"]),
        "G_Low":  self._down4096_to_256(src["GchannelLow"]),
        "G_High": self._down4096_to_256(src["GchannelHigh"]),
        "B_Low":  self._down4096_to_256(src["BchannelLow"]),
        "B_High": self._down4096_to_256(src["BchannelHigh"]),
    }

    panel, fr, my = self._get_ui_meta()
    X, contract = self._build_feature_matrix_W_checked(
        lut256, panel_text=panel, frame_rate=fr, model_year=my
    )

    # 2) 예측하고 통계 로그
    def _pred(payload):
        base = payload["linear_model"].predict(X).astype(np.float32)
        resid= payload["rf_residual"].predict(X).astype(np.float32)
        mu   = float(payload["target_scaler"]["mean"])
        sd   = float(payload["target_scaler"]["std"])
        return (base + resid) * sd + mu

    for comp in ("Gamma","Cx","Cy"):
        y = _pred(self.models_Y0_bundle[comp])
        logging.debug(f"[Predict/Debug] {comp}: mean={np.nanmean(y):.6g}, std={np.nanstd(y):.6g}, min={np.nanmin(y):.6g}, max={np.nanmax(y):.6g}")

    # 3) g=128 한 줄 피처 상세
    g = 128
    logging.debug(f"[Predict/Debug] g={g} feature row: {X[g,:].tolist()}")