        # 6) knot delta → per-gray LUT 후보 (12bit 스케일 유지)
        corr_RL = Phi @ dh_RL
        corr_GL = Phi @ dh_GL
        corr_BL = Phi @ dh_BL
        corr_RH = Phi @ dh_RH
        corr_GH = Phi @ dh_GH
        corr_BH = Phi @ dh_BH

        lut256_new = {
            "R_Low":  (lut256["R_Low"]  + corr_RL).astype(np.float32),
            "G_Low":  (lut256["G_Low"]  + corr_GL).astype(np.float32),
            "B_Low":  (lut256["B_Low"]  + corr_BL).astype(np.float32),
            "R_High": (lut256["R_High"] + corr_RH).astype(np.float32),
            "G_High": (lut256["G_High"] + corr_GH).astype(np.float32),
            "B_High": (lut256["B_High"] + corr_BH).astype(np.float32),
        }

        # =========================
        # ▼ NEW: 안전 후처리 파이프라인
        # =========================

        for ch in ("R","G","B"):
            Lk = f"{ch}_Low"
            Hk = f"{ch}_High"

            # --- (0) 엔드포인트 0/4095 선고정 (초기화용)
            #     아직 최종은 아니지만, 너무 이상하게 튄 경우 안정시키기
            lut256_new[Lk][0]   = 0.0
            lut256_new[Hk][0]   = 0.0
            lut256_new[Lk][255] = 4095.0
            lut256_new[Hk][255] = 4095.0

            # --- (1) Low/High 역전 금지 (1차 정리)
            low_fixed, high_fixed = self._fix_low_high_order(
                lut256_new[Lk], lut256_new[Hk]
            )

            # --- (2) 스무딩 + 단조 (고주파 톱니 제거)
            low_smooth  = self._smooth_and_monotone(low_fixed,  win=9)
            high_smooth = self._smooth_and_monotone(high_fixed, win=9)

            # --- (3) 감마(midpoint) 너무 어긋난 구간만 살짝 당김
            low_mid, high_mid = self._nudge_midpoint(
                low_smooth, high_smooth,
                max_err=3.0,   # 12bit scale에서 ±3카운트 넘게 어긋난 곳만
                strength=0.5   # 절반만 수정
            )

            # --- (4) 최종 안전화:
            #       - 다시 Low<=High
            #       - 단조 재보장
            #       - 0/255 고정
            #       - clip
            low_final, high_final = self._finalize_channel_pair_safely(
                low_mid, high_mid
            )

            lut256_new[Lk] = low_final
            lut256_new[Hk] = high_final

        # 이제 lut256_new[*] 는
        #  - 단조 증가
        #  - Low <= High
        #  - g=0은 0, g=255는 4095
        #  - 들쭉날쭉(tremor) 완화
        # =========================
        # ▲ NEW 파이프라인 끝
        # =========================

        # --------- (기존) 디버그 로그 호출 유지 -------------
        try:
            self._debug_log_knot_update(
                iter_idx=iter_idx,
                knots=knots,
                delta_h=delta_h,
                lut256_before=lut256_before,
                lut256_after=lut256_new,
            )
        except Exception:
            logging.exception("[CORR DEBUG] _debug_log_knot_update failed")
        # ----------------------------------------------------

        # 9) 256 → 4096 업샘플 (모든 채널), 정수화
        new_lut_4096 = {
            "RchannelLow":  self._up256_to_4096(lut256_new["R_Low"]),
            "GchannelLow":  self._up256_to_4096(lut256_new["G_Low"]),
            "BchannelLow":  self._up256_to_4096(lut256_new["B_Low"]),
            "RchannelHigh": self._up256_to_4096(lut256_new["R_High"]),
            "GchannelHigh": self._up256_to_4096(lut256_new["G_High"]),
            "BchannelHigh": self._up256_to_4096(lut256_new["B_High"]),
        }

        for k in new_lut_4096:
            new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)