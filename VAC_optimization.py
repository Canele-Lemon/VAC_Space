# (a) 예측 ON (W)
lut256_iter = {
    "R_Low": lut256["R_Low"], "G_Low": lut256["G_Low"], "B_Low": lut256["B_Low"],
    "R_High": high_R, "G_High": high_G, "B_High": high_B
}
y_pred = self._predict_Y0W_from_models(
    lut256_iter,
    panel_text=panel, frame_rate=fr, model_year=model_year
)

# ★ 디버그 덤프: 현재 반복/컨텍스트 태그로 저장
self._debug_dump_predicted_Y0W(
    y_pred,
    tag=f"iter{it}_{panel}_fr{fr}_my{int(model_year)%100:02d}",
    save_csv=True
)