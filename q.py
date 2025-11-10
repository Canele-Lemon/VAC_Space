for g in range(n_gray):
    idxLUT = int(mapLUT[g]) if 0 <= g < len(mapLUT) else -1

    row = {
        "gray": int(g),
        "LUT idx": idxLUT,  # 공통 LUT index
        "CORR": int(corr_flag[g]),
        "ΔCx": float(d_targets["Cx"][g]) if np.isfinite(d_targets["Cx"][g]) else np.nan,
        "ΔCy": float(d_targets["Cy"][g]) if np.isfinite(d_targets["Cy"][g]) else np.nan,
        "ΔGamma": float(d_targets["Gamma"][g]) if np.isfinite(d_targets["Gamma"][g]) else np.nan,
        "ΔR": float(dR_gray[g]) if np.isfinite(dR_gray[g]) else 0.0,
        "ΔG": float(dG_gray[g]) if np.isfinite(dG_gray[g]) else 0.0,
        "ΔB": float(dB_gray[g]) if np.isfinite(dB_gray[g]) else 0.0,
        # per-gray weight
        "wCx": float(wCx_gray[g]) if np.isfinite(wCx_gray[g]) else np.nan,
        "wCy": float(wCy_gray[g]) if np.isfinite(wCy_gray[g]) else np.nan,
        "wGamma": float(wG_gray[g]) if np.isfinite(wG_gray[g]) else np.nan,
    }

    if 0 <= idxLUT < len(RH0):
        row["R_before"] = float(RH0[idxLUT])
        row["R_after"]  = float(RH[idxLUT])
        row["G_before"] = float(GH0[idxLUT])
        row["G_after"]  = float(GH[idxLUT])
        row["B_before"] = float(BH0[idxLUT])
        row["B_after"]  = float(BH[idxLUT])
    else:
        row["R_before"] = np.nan
        row["R_after"]  = np.nan
        row["G_before"] = np.nan
        row["G_after"]  = np.nan
        row["B_before"] = np.nan
        row["B_after"]  = np.nan

    rows.append(row)