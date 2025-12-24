def _generate_predicted_vac_lut(
    self,
    base_vac_dict: dict,
    *,
    n_iters: int = 1,
    wG: float = 0.4,
    wC: float = 1.0,
    lambda_ridge: float = 1e-3,
    use_pattern_onehot: bool = False,
    patterns: tuple = ("W",),
    bypass_vac_info_pk: int = 1,
):
    """
    Returns
    -------
    vacparam_str : str | None
        ✅ TV에 바로 write 가능한 탭 포맷 문자열 (build_vacparam_std_format 결과)
    new_lut_tvkeys : dict | None
        ✅ TV 원 키명 그대로의 4096 LUT dict (plot/verify/로그용)
        {"RchannelLow":[...], "RchannelHigh":[...], ...}  # list[int] 권장
    debug_info : dict
    """
    debug_info = {"iters": [], "bypass_vac_info_pk": bypass_vac_info_pk}

    try:
        # --- (중간 로직은 사용자가 올린 그대로 유지) ---
        # 0~8단: jacobian/model check, idx_map, bypass fetch,
        # base/bypass 4096->256, meta, predict, jacobian update loop
        # ... your code ...

        # -----------------------------
        # 9) 256 -> 4096 upsample (High only)
        # -----------------------------
        # ✅ build_vacparam_std_format은 list/np-array 둘 다 받지만
        #    내부에서 arr.astype(int) 후 .tolist() 하므로 최종 list가 됨.
        #    여기서는 "int list" 형태로 넘기는게 가장 안전함.

        def _to_int_list4096(x) -> list:
            a = np.asarray(x).astype(np.int32)
            if a.size != 4096:
                raise ValueError(f"4096 LUT size mismatch: {a.size}")
            a = np.clip(a, 0, 4095)
            return a.tolist()

        # base Low는 그대로, High만 업데이트
        new_lut_tvkeys = {
            "RchannelLow":  _to_int_list4096(base_RL),  # base_RL: (4096,)
            "GchannelLow":  _to_int_list4096(base_GL),
            "BchannelLow":  _to_int_list4096(base_BL),
            "RchannelHigh": _to_int_list4096(np.round(self._up256_to_4096(high_R))),
            "GchannelHigh": _to_int_list4096(np.round(self._up256_to_4096(high_G))),
            "BchannelHigh": _to_int_list4096(np.round(self._up256_to_4096(high_B))),
        }

        # -----------------------------
        # 10) ✅ TV write용 "조립 문자열" 생성
        # -----------------------------
        vacparam_str = self.build_vacparam_std_format(
            base_vac_dict=base_vac_dict,
            new_lut_tvkeys=new_lut_tvkeys
        )

        # -----------------------------
        # 11) return
        # -----------------------------
        return vacparam_str, new_lut_tvkeys, debug_info

    except Exception:
        logging.exception("[PredictOpt] failed")
        return None, None, debug_info