# ...
yG[0] = np.nan; yG[255] = np.nan
out = {"Gamma": yG, "Cx": yCx, "Cy": yCy}

# (선택) 여기서도 덤프하고 싶으면:
# self._debug_dump_predicted_Y0W(out, tag="fast_path", save_csv=False)

return out