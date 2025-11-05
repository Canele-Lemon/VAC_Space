if pattern == 'white':
    is_on_session = (profile.ref_store is not None)
    is_fine_mode = getattr(self, "_fine_mode", False)

    # ★ ON 세션의 0gray(main) 휘도 저장 → 이후 정규화에 사용
    if is_on_session:
        ref_store = profile.ref_store
        # main role 기준으로 0gray 휘도 사용
        lv0_main, _, _ = store['gamma']['main']['white'].get(0, (np.nan, np.nan, np.nan))
        if np.isfinite(lv0_main):
            self._on_lv0_current = float(lv0_main)