def _update_lut_chart_and_table(self, lut_dict):
    try:
        required = ["R_Low","R_High","G_Low","G_High","B_Low","B_High"]
        for k in required:
            if k not in lut_dict:
                logging.error(f"missing key: {k}")
                return
            if len(lut_dict[k]) != 4096:
                logging.error(f"invalid length for {k}: {len(lut_dict[k])} (expected 4096)")
                return

        # 테이블만 갱신
        df = pd.DataFrame({
            "R_Low":  lut_dict["R_Low"],
            "R_High": lut_dict["R_High"],
            "G_Low":  lut_dict["G_Low"],
            "G_High": lut_dict["G_High"],
            "B_Low":  lut_dict["B_Low"],
            "B_High": lut_dict["B_High"],
        })
        self.update_rgbchannel_table(df, self.ui.vac_table_rbgLUT_4)

        # 차트는 새 데이터로 리셋 후 재그림
        self.vac_optimization_lut_chart.reset_and_plot(lut_dict)

    except Exception as e:
        logging.exception(e)
        
        
def _after_read_back(vac_dict_after):
    if not vac_dict_after:
        logging.error("보정 후 VAC 재읽기 실패")
        return
    self._vac_dict_cache = vac_dict_after
    lut_dict_plot = {k.replace("channel","_"): v
                     for k, v in vac_dict_after.items() if "channel" in k}
    # ✅ 테이블 + 차트 동시 갱신(신규 방식)
    self._update_lut_chart_and_table(lut_dict_plot)
    # 여기서 다음 측정 세션 이어가기...