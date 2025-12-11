    def build_XYdataset_for_jacobian_g(self, component='dGamma'):
        """
        자코비안/보정용:
        - X: raw ΔLUT(High 3채널) + meta + gray_norm + LUT index
        - y: dGamma / dCx / dCy (White 패턴, target - ref)
        """
        assert component in ('dGamma','dCx','dCy')

        # 자코비안은 High 채널만
        jac_channels = ('R_High', 'G_High', 'B_High')

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
            Yd  = s["Y"]  # {"Y0": {...}, ...}

            p = 'W'
            y_vec = Yd['Y0'][p][component]  # (256,)
            for g in range(256):
                y_val = y_vec[g]
                if not np.isfinite(y_val):
                    continue
                feat_row = self._build_features_for_gray(
                    X_dict=Xd,
                    gray=g,
                    channels=jac_channels,
                )
                X_rows.append(feat_row)
                y_vals.append(float(y_val))
                groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups
        
    def _build_XY_Y0(
        self,
        component: str = "dGamma",
        channels=('R_Low','R_High','G_Low','G_High','B_Low','B_High'),
        patterns=('W',),
    ):
        """
        Y0 예측용(X→dGamma/dCx/dCy) per-gray 데이터셋.
        - X: ΔLUT(지정 채널) + meta + gray_norm + LUT index
        - y: 선택한 component (dGamma/dCx/dCy), 지정된 패턴들(W/R/G/B)
        """
        assert component in ('dGamma', 'dCx', 'dCy')

        X_rows, y_vals, groups = [], [], []

        for s in self.samples:
            pk  = s["pk"]
            Xd  = s["X"]  # {"lut_delta_raw":..., "meta":..., "mapping_j":...}
            Yd  = s["Y"]  # {"Y0": {...}, "Y1": {...}, "Y2": {...}}

            for p in patterns:
                if "Y0" not in Yd or p not in Yd["Y0"]:
                    continue
                y_vec = Yd["Y0"][p][component]  # (256,)
                for g in range(256):
                    y_val = y_vec[g]
                    if not np.isfinite(y_val):
                        continue
                    feat_row = self._build_features_for_gray(
                        X_dict=Xd,
                        gray=g,
                        channels=channels,
                    )
                    X_rows.append(feat_row)
                    y_vals.append(float(y_val))
                    groups.append(pk)

        X_mat = np.vstack(X_rows).astype(np.float32) if X_rows else np.empty((0,0), np.float32)
        y_vec = np.asarray(y_vals, dtype=np.float32)
        groups = np.asarray(groups, dtype=np.int64)
        return X_mat, y_vec, groups
        
    def build_XY_dataset(
        self,
        target: str,
        component: str | None = None,
        channels=None,
        patterns=('W',),
    ):
        """
        통합 XY 데이터셋 빌더.

        Parameters
        ----------
        target : {'Y0', 'Y1', 'Y2', 'jacobian'}
            어떤 타겟을 예측할지 선택.
        component : str | None
            - target='Y0' 또는 'jacobian' 일 때: {'dGamma','dCx','dCy'}
            - target='Y1','Y2' 에서는 사용 안 함(또는 향후 확장 용도).
        channels : tuple[str] | None
            X에 사용할 LUT 채널 리스트.
            - target='jacobian' 인 경우: None이면 ('R_High','G_High','B_High')
            - target='Y0' 인 경우: None이면 ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
        patterns : tuple[str]
            사용할 패턴 (지금은 보통 ('W',) 로 쓰는 걸 가정)

        Returns
        -------
        X_mat : np.ndarray
        y_vec : np.ndarray
        groups : np.ndarray
        """
        target = target.lower()

        if target == "jacobian":
            # 자코비안: High 3채널 고정, 기존 메서드 재사용
            if component is None:
                component = "dGamma"
            return self.build_XYdataset_for_jacobian_g(component=component)

        if target == "y0":
            if component is None:
                raise ValueError("target='Y0'일 때 component('dGamma','dCx','dCy')가 필요합니다.")
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY_Y0(component=component, channels=channels, patterns=patterns)

        if target == "y1":
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY_Y1(channels=channels, patterns=patterns)

        if target == "y2":
            if channels is None:
                channels = ('R_Low','R_High','G_Low','G_High','B_Low','B_High')
            return self._build_XY_Y2(channels=channels)

        raise ValueError(f"Unknown target='{target}'. (지원: 'jacobian','Y0','Y1','Y2')")