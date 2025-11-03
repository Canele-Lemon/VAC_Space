# ===== [추가] 클래스 init에서 feature 채널/패턴/컴포넌트 제어 옵션 =====
class VACDataset(Dataset):
    def __init__(self, pk_list, 
                 y0_component='dGamma',            # 'dGamma' | 'dCx' | 'dCy'
                 patterns=('W',),                  # White만
                 feature_channels=('R_High','G_High','B_High'),  # Low 제외
                 reference_pk=None                 # 필요시 ref PK 지정 (없으면 VACOutputBuilder 기본값)
                 ):
        self.pk_list = list(pk_list)
        self.y0_component = y0_component
        self.patterns = patterns
        self.feature_channels = feature_channels
        self.reference_pk = reference_pk
        self.samples = []
        self._collect()

    def _collect(self):
        for pk in self.pk_list:
            x_builder = VACInputBuilder(pk)
            # X는 ΔLUT(정규화 256포인트) — 필요 시 raw-mapping 버전으로 교체 가능
            X = x_builder.prepare_X_delta()

            # reference_pk를 외부에서 지정하면 그걸로 Y0(d*) 계산
            if self.reference_pk is None:
                y_builder = VACOutputBuilder(pk)  # 내부 기본 ref 사용
            else:
                y_builder = VACOutputBuilder(pk, reference_pk=self.reference_pk)

            Y = y_builder.prepare_Y(y1_patterns=('W',))

            self.samples.append({"pk": pk, "X": X, "Y": Y})

    # ===== [수정] 피처 생성에서 채널 리스트를 옵션으로 받도록 =====
    def _build_features_for_gray(self, X_dict, gray: int, add_pattern: str | None = None) -> np.ndarray:
        lut = X_dict["lut"]
        meta = X_dict["meta"]

        row = []
        # 여기서 self.feature_channels만 사용 (Low 고정이므로 High만)
        for ch in self.feature_channels:
            row.append(float(lut[ch][gray]))

        # 메타
        panel_vec = np.asarray(meta["panel_maker"], dtype=np.float32).tolist()
        row.extend(panel_vec)
        row.append(float(meta["frame_rate"]))
        row.append(float(meta["model_year"]))
        row.append(gray / 255.0)

        if add_pattern is not None:
            pat_idx = _PATTERN_LIST.index(add_pattern) if add_pattern in _PATTERN_LIST else -1
            row.extend(_onehot(pat_idx, len(_PATTERN_LIST)).tolist())

        return np.asarray(row, dtype=np.float32)

    # ===== [추가] White + Y0(d*) 전용 빌더 =====
    def build_white_y0_delta(self):
        """
        White 패턴만, y = dGamma/dCx/dCy (선택) 단일 스칼라 회귀용.
        X는 (High 채널만 + meta + gray_norm + pattern onehot)
        """
        component = self.y0_component
        assert component in ('dGamma','dCx','dCy')

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # ΔLUT dict
            Yd  = s["Y"]  # ΔY dict
            # White만
            p = 'W'
            y_vec = Yd['Y0'][p][component]  # (256,)
            for g in range(256):
                y_val = y_vec[g]
                if not np.isfinite(y_val):
                    continue
                feat_row = self._build_features_for_gray(X_dict=Xd, gray=g, add_pattern=p)
                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups

# ===== [추가] 디버그 프린트: White + High-only 구성을 눈으로 확인 =====
def debug_preview_white_y0(pk_list, 
                           component='dGamma', 
                           feature_channels=('R_High','G_High','B_High'), 
                           reference_pk=None, 
                           max_print=8):
    ds = VACDataset(pk_list=pk_list, 
                    y0_component=component, 
                    patterns=('W',), 
                    feature_channels=feature_channels,
                    reference_pk=reference_pk)

    X_mat, y_vec, groups = ds.build_white_y0_delta()

    print("[DEBUG] WHITE-only Y0(Δ) dataset")
    print("  component     :", component)
    print("  feature_ch    :", feature_channels)
    print("  X_mat shape   :", X_mat.shape)
    print("  y_vec shape   :", y_vec.shape)
    print("  groups shape  :", groups.shape)

    if X_mat.shape[0] == 0:
        print("  -> No valid samples.")
        return

    panel_len = len(ds.samples[0]["X"]["meta"]["panel_maker"])
    # feature layout:
    # [ΔR_High, ΔG_High, ΔB_High]  (len(feature_channels)=3)
    # + panel_onehot(K) + frame_rate + model_year + gray_norm + pattern_onehot(4)

    F = len(feature_channels)
    for i in range(min(max_print, X_mat.shape[0])):
        feat = X_mat[i]
        y    = y_vec[i]
        pk   = groups[i]

        delta_lut = feat[:F]
        idx = F
        panel_oh = feat[idx: idx + panel_len]; idx += panel_len
        frame_rate = feat[idx]; idx += 1
        model_year = feat[idx]; idx += 1
        gray_norm  = feat[idx]; idx += 1
        pattern_oh = feat[idx: idx+4]
        est_gray   = gray_norm * 255.0
        pat_idx    = int(np.argmax(pattern_oh))
        pat_name   = _PATTERN_LIST[pat_idx] if 0 <= pat_idx < 4 else 'NA'

        print(f"\n--- sample {i} ---")
        print(f"pk: {pk} / pattern: {pat_name} / gray≈{est_gray:.1f}")
        print(f"ΔLUT({feature_channels}): {delta_lut}")
        print(f"panel_onehot : {panel_oh}")
        print(f"frame_rate   : {frame_rate}")
        print(f"model_year   : {model_year}")
        print(f"y (target {component}) : {y}")

if __name__ == "__main__":
    # 예시: dGamma, White, High-only, ref=2582 로 비교
    debug_preview_white_y0(
        pk_list=[2635], 
        component='dGamma',
        feature_channels=('R_High','G_High','B_High'),
        reference_pk=2582,
        max_print=10
    )