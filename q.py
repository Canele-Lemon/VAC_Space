        # 7) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
        new_lut_4096 = {
            "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
            "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
            "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
            "RchannelHigh": RH,
            "GchannelHigh": GH,
            "BchannelHigh": BH,
        }
        for k in new_lut_4096:
            arr = np.asarray(new_lut_4096[k], dtype=np.float32)
            # 혹시라도 남아있을 NaN을 안전하게 제거 (0으로)
            arr = np.nan_to_num(arr, nan=0.0)
            new_lut_4096[k] = np.clip(np.round(arr), 0, 4095).astype(np.uint16)

        # 🔵 6') {iter_idx}회차 보정 결과를 DataFrame으로 구성
        rows = []
        n_gray = 256
        for g in range(n_gray):
            # LUT index는 R 채널 기준으로 사용 (필요하면 R/G/B 따로 뽑아도 됨)
            idxR = int(mapR[g]) if 0 <= g < len(mapR) else -1
            idxG = int(mapG[g]) if 0 <= g < len(mapG) else -1
            idxB = int(mapB[g]) if 0 <= g < len(mapB) else -1

            row = {
                "gray": int(g),
                "LUT idx": idxR,
                "CORR": int(corr_flag[g]),  # 1: 보정 필요(gray ∈ NG), 0: OK
                "ΔCx": float(d_targets["Cx"][g]) if np.isfinite(d_targets["Cx"][g]) else np.nan,
                "ΔCy": float(d_targets["Cy"][g]) if np.isfinite(d_targets["Cy"][g]) else np.nan,
                "ΔGamma": float(d_targets["Gamma"][g]) if np.isfinite(d_targets["Gamma"][g]) else np.nan,
                "ΔR": float(dR_gray[g]) if np.isfinite(dR_gray[g]) else 0.0,
                "ΔG": float(dG_gray[g]) if np.isfinite(dG_gray[g]) else 0.0,
                "ΔB": float(dB_gray[g]) if np.isfinite(dB_gray[g]) else 0.0,
            }

            # R/G/B before/after
            if 0 <= idxR < len(RH0):
                row["R_before"] = float(RH0[idxR])
                row["R_after"]  = float(RH[idxR])
            else:
                row["R_before"] = np.nan
                row["R_after"]  = np.nan

            if 0 <= idxG < len(GH0):
                row["G_before"] = float(GH0[idxG])
                row["G_after"]  = float(GH[idxG])
            else:
                row["G_before"] = np.nan
                row["G_after"]  = np.nan

            if 0 <= idxB < len(BH0):
                row["B_before"] = float(BH0[idxB])
                row["B_after"]  = float(BH[idxB])
            else:
                row["B_before"] = np.nan
                row["B_after"]  = np.nan

            rows.append(row)

        df_corr = pd.DataFrame(rows, columns=[
            "gray", "LUT idx", "CORR",
            "ΔCx", "ΔCy", "ΔGamma",
            "ΔR", "ΔG", "ΔB",
            "R_before", "R_after",
            "G_before", "G_after",
            "B_before", "B_after",
        ])

        # 나중에 디버깅/저장을 위해 객체에 들고 있기
        self._last_batch_corr_df = df_corr

        # 로그 한 번만 찍기
        logging.info(
            f"[Batch Correction] {iter_idx}회차 보정 결과:\n"
            + df_corr.to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )