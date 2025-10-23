ui로부터 메타 정보를 불러와야 하잖아요.

            # 4) UI 메타
            panel = self.ui.vac_cmb_PanelMaker.currentText().strip()
            fr    = float(self.ui.vac_cmb_FrameRate.currentText().strip())
            # model_year는 알 수 없으면 0으로
            model_year = float(getattr(self, "current_model_year", 0.0))

여기서 패널은 vac_cmb_PanelMaker 라는 qcombobox, fr는 vac_cmb_FrameRate라는 qcombobox의 text는 맞는데 "Hz"라는 어미가 붙어있어 이걸 떼 주고 float해야 해요. 그리고 model year은 vac_cmb_ModelYear라는 qcombobox의 텍스트인데 이것도 "Y"가 붙어있어서 이걸 떼 주고 float 해야 합니다.
