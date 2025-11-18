    def build_full_LUT_dataframe(self):
        """
        현재 테이블의 256개 High knot 값 + Low LUT(4096) + 보간 결과를 담아
        GrayLevel_window, R_Low, R_High, ... B_High 총 4096행 DataFrame 반환
        """
        # --- Low LUT ---
        Rl, Gl, Bl = self.load_low_lut_4096()
        j_axis = np.arange(4096)

        # --- High knot 읽기 ---
        Gray12 = []
        Rvals = []
        Gvals = []
        Bvals = []

        for r in range(256):
            item_g12 = self.model.item(r, 1)
            item_r   = self.model.item(r, 2)
            item_g   = self.model.item(r, 3)
            item_b   = self.model.item(r, 4)

            if (item_g12 is None or item_g12.text().strip() == "" or
                item_r   is None or item_r.text().strip() == "" or
                item_g   is None or item_g.text().strip() == "" or
                item_b   is None or item_b.text().strip() == ""):
                # High LUT 불완전 → None 반환
                return None

            Gray12.append(float(item_g12.text()))
            Rvals.append(float(item_r.text()))
            Gvals.append(float(item_g.text()))
            Bvals.append(float(item_b.text()))

        # numpy 변환 & sort
        Gray12 = np.array(Gray12, float)
        Rvals = np.array(Rvals, float)
        Gvals = np.array(Gvals, float)
        Bvals = np.array(Bvals, float)

        idx = np.argsort(Gray12)
        Gray12 = Gray12[idx]
        Rvals = Rvals[idx]
        Gvals = Gvals[idx]
        Bvals = Bvals[idx]

        # 보간
        R_full = np.interp(j_axis, Gray12, Rvals)
        G_full = np.interp(j_axis, Gray12, Gvals)
        B_full = np.interp(j_axis, Gray12, Bvals)

        # DataFrame 구성
        LUT = pd.DataFrame({
            "GrayLevel_window": j_axis,
            "R_Low":  Rl,
            "R_High": R_full,
            "G_Low":  Gl,
            "G_High": G_full,
            "B_Low":  Bl,
            "B_High": B_full,
        })

        for c in ["R_Low","R_High","G_Low","G_High","B_Low","B_High"]:
            LUT[c] = np.rint(LUT[c]).astype(int)
    
        return LUT

이 코드가 현재 tableView_LUT 테이블의 256knots 값들에 대해 interpolation 하는 코드인가요?
