def _debug_log_knot_update(self, iter_idx, knots, delta_h, lut256_before, lut256_after):
    """
    iter_idx        : 현재 iteration 번호 (1, 2, ...)
    knots           : self._jac_artifacts["knots"]  # 길이 K, 예: [0,8,16,...,255]
    delta_h         : (6K,) 이번 iteration에서 solve한 Δh
    lut256_before   : dict of 6채널 256길이 LUT (보정 전, float32)
    lut256_after    : dict of 6채널 256길이 LUT (보정 후, float32)

    이걸 로그에 예쁘게 찍어 분석용으로 쓸 수 있게 해 준다.
    """
    try:
        K = len(knots)
        # 채널 분해
        dh_RL = delta_h[0*K : 1*K]
        dh_GL = delta_h[1*K : 2*K]
        dh_BL = delta_h[2*K : 3*K]
        dh_RH = delta_h[3*K : 4*K]
        dh_GH = delta_h[4*K : 5*K]
        dh_BH = delta_h[5*K : 6*K]

        def _summ(ch_name, dh_vec):
            # dh_vec 길이 K
            # 상위 몇 개만 큰 변화 순으로 보여주면 어디가 움직였는지 직관적으로 파악 가능
            mag = np.abs(dh_vec)
            top_idx = np.argsort(mag)[::-1][:5]  # 변화량 큰 상위 5개 knot
            msg_lines = [f"    {ch_name} top5 |knot(gray)->Δh|:"]
            for i in top_idx:
                msg_lines.append(
                    f"      knot#{i:02d} (gray≈{knots[i]:3d}) : Δh={dh_vec[i]:+.4f}"
                )
            return "\n".join(msg_lines)

        logging.info("======== [CORR DEBUG] Iter %d Knot Δh ========\n%s\n%s\n%s\n%s\n%s\n%s",
            iter_idx,
            _summ("R_Low ", dh_RL),
            _summ("G_Low ", dh_GL),
            _summ("B_Low ", dh_BL),
            _summ("R_High", dh_RH),
            _summ("G_High", dh_GH),
            _summ("B_High", dh_BH),
        )

        # LUT 전/후 차이도 간단 비교 (예: High 채널만 대표로)
        def _lut_diff_stats(name):
            before = np.asarray(lut256_before[name], dtype=np.float32)
            after  = np.asarray(lut256_after[name],  dtype=np.float32)
            diff   = after - before
            return (float(np.min(diff)),
                    float(np.max(diff)),
                    float(np.mean(diff)),
                    float(np.std(diff)))

        for ch in ["R_Low","G_Low","B_Low","R_High","G_High","B_High"]:
            dmin, dmax, dmean, dstd = _lut_diff_stats(ch)
            logging.debug(
                "[CORR DEBUG] Iter %d %s LUT256 delta stats: "
                "min=%+.4f max=%+.4f mean=%+.4f std=%.4f",
                iter_idx, ch, dmin, dmax, dmean, dstd
            )

    except Exception:
        logging.exception("[CORR DEBUG] knot update logging failed")