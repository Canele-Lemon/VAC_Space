    def _build_batch_corr_df(
        self,
        iter_idx: int,
        d_targets: dict,
        dR_gray: np.ndarray,
        dG_gray: np.ndarray,
        dB_gray: np.ndarray,
        corr_flag: np.ndarray,
        mapR: np.ndarray,
        mapG: np.ndarray,
        mapB: np.ndarray,
        RH0: np.ndarray, GH0: np.ndarray, BH0: np.ndarray,
        RH:  np.ndarray, GH:  np.ndarray, BH:  np.ndarray,
    ):
        """
        회차별 보정 결과 DF 생성 + 로그 + CSV 저장
        컬럼:
        gray | LUT idx | CORR | ΔCx | ΔCy | ΔGamma | ΔR | ΔG | ΔB |
        R_before | R_after | G_before | G_after | B_before | B_after
        """
        rows = []
        n_gray = 256

        for g in range(n_gray):
            idxR = int(mapR[g]) if 0 <= g < len(mapR) else -1
            idxG = int(mapG[g]) if 0 <= g < len(mapG) else -1
            idxB = int(mapB[g]) if 0 <= g < len(mapB) else -1

            row = {
                "gray": int(g),
                "LUT idx": idxR,  # 기준으로 R High 인덱스를 사용
                "CORR": int(corr_flag[g]),  # 1: 이 gray는 이번 회차 보정 대상(NG), 0: OK
                "ΔCx": float(d_targets["Cx"][g]) if np.isfinite(d_targets["Cx"][g]) else np.nan,
                "ΔCy": float(d_targets["Cy"][g]) if np.isfinite(d_targets["Cy"][g]) else np.nan,
                "ΔGamma": float(d_targets["Gamma"][g]) if np.isfinite(d_targets["Gamma"][g]) else np.nan,
                "ΔR": float(dR_gray[g]) if np.isfinite(dR_gray[g]) else 0.0,
                "ΔG": float(dG_gray[g]) if np.isfinite(dG_gray[g]) else 0.0,
                "ΔB": float(dB_gray[g]) if np.isfinite(dB_gray[g]) else 0.0,
            }

            # R before/after
            if 0 <= idxR < len(RH0):
                row["R_before"] = float(RH0[idxR])
                row["R_after"]  = float(RH[idxR])
            else:
                row["R_before"] = np.nan
                row["R_after"]  = np.nan

            # G
            if 0 <= idxG < len(GH0):
                row["G_before"] = float(GH0[idxG])
                row["G_after"]  = float(GH[idxG])
            else:
                row["G_before"] = np.nan
                row["G_after"]  = np.nan

            # B
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

        self._last_batch_corr_df = df_corr
        
        return df_corr

현재 df_corr는 위처럼 생성되는데 
1. mapR(or G or B)는 무엇인가요?
2. 이제 각 gray 마다 Cx, Cy, Gamma 가중치가 달라지는데 이것도 df에 반영해주세요
