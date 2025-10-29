def _run_correction_iteration(self, iter_idx, max_iters=2):
    """
    새 로컬 보정 알고리즘:
    1) OFF vs ON 차이(d_targets)에서 spec-out gray들만 고른다.
    2) 각 gray g에 대해 High LUT을 색 보정 방향으로 한 스텝(작게) 이동.
    3) 이동한 만큼 Low LUT을 역보상해서 (Low+High)/2 가 원래 목표 밝기(=g의 12bit 스케일)에 맞게 유지.
       -> 감마 안정
    4) 주변 ±N gray로 스무딩.
    5) monotone/clip, 업샘플, TV write, 재측정.
    """

    logging.info(f"[CORR-LOCAL] iteration {iter_idx} start (local nudge w/ avg constraint)")
    self._step_start(2)

    # 1) 현재 TV LUT (4096포인트) 가져오기
    if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
        logging.warning("[CORR-LOCAL] LUT 캐시 없음 → 직전 읽기 결과가 필요합니다.")
        return None
    vac_dict = self._vac_dict_cache  # TV에서 방금 읽어온 VAC JSON dict

    # 2) 4096 → 256 다운샘플 (12bit 스케일 그대로)
    lut256 = {
        "R_Low":  self._down4096_to_256(vac_dict["RchannelLow"]),
        "G_Low":  self._down4096_to_256(vac_dict["GchannelLow"]),
        "B_Low":  self._down4096_to_256(vac_dict["BchannelLow"]),
        "R_High": self._down4096_to_256(vac_dict["RchannelHigh"]),
        "G_High": self._down4096_to_256(vac_dict["GchannelHigh"]),
        "B_High": self._down4096_to_256(vac_dict["BchannelHigh"]),
    }

    # 3) spec-out 타깃 계산 (OFF vs 최신 ON 비교)
    d_targets = self._build_delta_targets_from_stores(self._off_store, self._on_store)
    # d_targets["Cx"][g] = ON - OFF (Cx 차이), 등등
    # 우리는 "이걸 0으로 만들고 싶다"가 목표.

    thr_c     = 0.003   # chromaticity spec
    thr_gamma = 0.05    # gamma spec (좀 느슨)
    # 작은 오차는 굳이 건드리지 않도록 0으로 마스킹
    for g in range(256):
        if (abs(d_targets["Cx"][g])    <= thr_c and
            abs(d_targets["Cy"][g])    <= thr_c and
            abs(d_targets["Gamma"][g]) <= thr_gamma):
            d_targets["Cx"][g]    = 0.0
            d_targets["Cy"][g]    = 0.0
            d_targets["Gamma"][g] = 0.0

    # 4) 어떤 gray를 "문제 구간"으로 볼 것인가?
    bad_mask = (
        (np.abs(d_targets["Cx"])    > thr_c) |
        (np.abs(d_targets["Cy"])    > thr_c) |
        (np.abs(d_targets["Gamma"]) > thr_gamma)
    )
    bad_grays = np.where(bad_mask)[0].tolist()
    logging.info(f"[CORR-LOCAL] bad_grays count={len(bad_grays)} (e.g. {bad_grays[:10]})")

    if len(bad_grays) == 0:
        logging.info("[CORR-LOCAL] 모든 포인트가 스펙 근처 → 더 조정할 게 없음")
        # 그래도 측정은 다시 돌려서 spec check 진행 루틴 태우기
        bad_grays = []

    # 5) local nudge 파라미터
    # step_size: High LUT을 얼마만큼 한 번에 움직일지 (12bit count 단위)
    step_size_color = 2.0     # 색좌표용 스텝 (작게!)
    step_size_gamma = 1.0     # 감마 보정쪽(평균 유지 틀어졌을 때) 스텝
    smooth_span     = 4       # 주변 ±4 gray까지 삼각 스무딩
    def _gray_to_12bit(g):
        return (g * 4095.0) / 255.0

    # 새 LUT 256버전 복사본 (float로 계속 작업)
    lut256_new = {k: v.copy().astype(np.float32) for k,v in lut256.items()}

    # 6) 각 bad gray에 대해 조정
    # 아이디어:
    #   - 색좌표 에러(Cx, Cy)는 주로 RGB 비율 문제라고 가정 → High 채널을 조정
    #   - 감마 에러(Gamma)도 크면 평균 (Low+High)/2 가 목표와 어긋났다고 보고 보정
    #
    # 여기선 single-channel Jacobian 없이 간단한 휴리스틱:
    #   Cx/Cy가 (+) 방향이면 R/G/B 중 어디를 늘려야 할지 정확히 모르지만
    #   우선은 전 채널 High를 소량 같은 방향으로 미는 대신,
    #   Gamma 에러도 같이 보면 됩니다.
    #
    # 현실적으로는 채널별 민감도를 학습해 sign을 정하는 게 더 좋지만,
    # 지금은 프레임만 먼저 만들어 드립니다.
    #
    # pseudo rule:
    #   - 색차가 크면 전체 High를 약간 내려서(또는 올려서) 색 온도를 움직이는 대신
    #     우선은 luminance 기반 제어만 틀어지지 않게 하는 프레임을 제공합니다.
    #
    # 사용자는 여기서 R/G/B별 방향결정 로직만 넣어주면 됩니다 (TODO 영역 표시).

    for g in bad_grays:
        tgt_mid = _gray_to_12bit(g)  # 우리가 유지하고 싶은 평균 (Low+High)/2
        # 현재 평균이 얼마나 틀어졌는지 (감마 관점)
        # 평균 = (R_Low+R_High)/2 이런 식으로 채널별로 다르지만,
        # 우선 화이트 관점으로 RGB를 같이 본다고 가정해서 각 채널별 평균을 보고 전체로 평균
        cur_avg_list = []
        for ch in ("R","G","B"):
            cur_L = lut256_new[f"{ch}_Low"][g]
            cur_H = lut256_new[f"{ch}_High"][g]
            cur_avg_list.append(0.5*(cur_L+cur_H))
        cur_avg = float(np.mean(cur_avg_list))
        gamma_err = cur_avg - tgt_mid  # +면 너무 밝다(평균 높다)

        # 색 에러 크기 (white/main 기준)
        color_err_mag = np.hypot(d_targets["Cx"][g], d_targets["Cy"][g])

        # ---------------------------
        # 결정 1: 감마쪽 보정 필요?
        # ---------------------------
        if abs(d_targets["Gamma"][g]) > thr_gamma or abs(gamma_err) > 2.0:
            # 평균이 tgt_mid보다 높으면 (너무 밝으면) 전체 High/Low를 내려서 평균 맞추는 쪽
            # 평균이 낮으면 전체 High/Low를 올리는 쪽
            sign_gamma = -1.0 if gamma_err > 0 else +1.0  # 평균이 높으면 내려야 하므로 -
            d_gamma_step = sign_gamma * step_size_gamma

            # 모든 채널에 같은 보정량 적용해서 밝기 맞추는 느낌
            for ch in ("R","G","B"):
                H_old = lut256_new[f"{ch}_High"][g]
                L_old = lut256_new[f"{ch}_Low"][g]

                H_new = H_old + d_gamma_step
                # 평균 유지 타깃: (L_new + H_new)/2 == tgt_mid
                L_new = 2.0*tgt_mid - H_new

                lut256_new[f"{ch}_High"][g] = H_new
                lut256_new[f"{ch}_Low"][g]  = L_new

        # ---------------------------
        # 결정 2: 색좌표쪽 보정 필요?
        # ---------------------------
        if color_err_mag > thr_c:
            # TODO: 채널별 방향성 결정
            # 지금은 placeholder로 "전 채널 High를 같은 방향으로 살짝 조정"만 넣고,
            # 그에 맞춰 Low를 역보상해서 평균(tgt_mid)은 다시 맞춘다.
            #
            # 색좌표가 Cx/Cy에서 어떤 방향으로 틀어졌는지에 따라
            #   예: Cx/Cy가 '푸르다' -> Blue High 줄이고 Red High 올리고 ...
            # 이런 로직을 넣어야 하는데, 아직 우리한테 그 맵핑룰(민감도)은 없으니
            # 여기서는 단일 부호(sign_color)를 0 으로 두고 패스하겠습니다.
            #
            sign_color_R = 0.0
            sign_color_G = 0.0
            sign_color_B = 0.0

            # 만약 나중에 규칙을 넣는다면 예를 들어:
            #   - dCx>0 and dCy>0 => (화이트 좌표가 더 red-ish/yellow-ish?) => R/G High 조금 감소 ...
            #   이런 식으로 채널별 sign 설정.
            #
            # 아래 구조만 만들어 둘게요.
            for ch, sign_ch in (("R",sign_color_R),
                                ("G",sign_color_G),
                                ("B",sign_color_B)):
                if sign_ch == 0.0:
                    continue
                H_old = lut256_new[f"{ch}_High"][g]
                L_old = lut256_new[f"{ch}_Low"][g]

                H_new = H_old + (step_size_color * sign_ch)
                L_new = 2.0*tgt_mid - H_new

                lut256_new[f"{ch}_High"][g] = H_new
                lut256_new[f"{ch}_Low"][g]  = L_new

    # 7) 스무딩: 각 bad gray 주변 ±smooth_span 에 걸쳐 선형 보간으로 부드럽게 퍼뜨리기
    for g in bad_grays:
        for gg in range(g - smooth_span, g + smooth_span + 1):
            if gg < 0 or gg >= 256: 
                continue
            dist = abs(gg - g)
            w = max(0.0, 1.0 - dist/float(smooth_span))  # 삼각 커널 1→0
            if w <= 0.0:
                continue
            for ch in ("R","G","B"):
                # 현재 gg 값과 g 값 사이를 w 비율만큼 섞어서 gg 쪽을 g에 더 가깝게 끌어온다
                for hl in ("Low","High"):
                    key = f"{ch}_{hl}"
                    base_val = lut256[key][gg]         # 원래값
                    target_val = lut256_new[key][g]    # g에서 조정한 값
                    blended = (1.0 - w)*lut256_new[key][gg] + w*target_val
                    # 위 한 줄만 써도 되지만 base_val 안 쓰면 톱니 조금 남을 수도 있음.
                    # 필요하면 base_val 대신 lut256_new[key][gg] 초기값 쓸 수도 있음.
                    lut256_new[key][gg] = blended

    # 8) monotone 및 clip(0..4095)
    for ch in ("R","G","B"):
        for hl in ("Low","High"):
            arr = lut256_new[f"{ch}_{hl}"]
            self._enforce_monotone(arr)
            lut256_new[f"{ch}_{hl}"] = np.clip(arr, 0, 4095)

    # 9) 256 → 4096 업샘플, uint16 변환
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

    # 10) UI용 차트/테이블 갱신
    lut_dict_plot = {
        "R_Low":  new_lut_4096["RchannelLow"],
        "R_High": new_lut_4096["RchannelHigh"],
        "G_Low":  new_lut_4096["GchannelLow"],
        "G_High": new_lut_4096["GchannelHigh"],
        "B_Low":  new_lut_4096["BchannelLow"],
        "B_High": new_lut_4096["BchannelHigh"],
    }
    self._update_lut_chart_and_table(lut_dict_plot)
    self._step_done(2)

    # 11) VAC JSON 다시 조립해서 TV write → 읽기 → 재측정 (하위 로직은 기존 그대로)
    def _after_write(ok, msg):
        logging.info(f"[VAC Write] {msg}")
        if not ok:
            return
        logging.info("[CORR-LOCAL] 보정 LUT TV Reading 시작")
        self._read_vac_from_tv(_after_read_back)

    def _after_read_back(vac_dict_after):
        if not vac_dict_after:
            logging.error("[CORR-LOCAL] 보정 후 VAC 재읽기 실패")
            return
        logging.info("[CORR-LOCAL] 보정 LUT TV Reading 완료")
        self._step_done(3)

        # 캐시 최신화
        self._vac_dict_cache = vac_dict_after

        # 차트 초기화 (ON 시리즈만 리셋)
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        # 다시 측정 세션
        profile_corr = SessionProfile(
            legend_text=f"CORR #{iter_idx}",
            cie_label=None,
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7, "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

        def _after_corr(store_corr):
            self._step_done(4)
            self._on_store = store_corr

            self._step_start(5)
            self._spec_thread = SpecEvalThread(
                self._off_store, self._on_store,
                thr_gamma=0.05, thr_c=0.003, parent=self
            )
            self._spec_thread.finished.connect(
                lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx, max_iters)
            )
            self._spec_thread.start()

        logging.info("[CORR-LOCAL] 보정 LUT 기준 측정 시작")
        self._step_start(4)
        self.start_viewing_angle_session(
            profile=profile_corr,
            gray_levels=getattr(op, "gray_levels_256", list(range(256))),
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000,
            cs_settle_ms=1000,
            on_done=_after_corr
        )

    logging.info(f"[CORR-LOCAL] LUT {iter_idx}차 TV Writing 시작")

    vac_write_json = self.build_vacparam_std_format(
        base_vac_dict=self._vac_dict_cache,
        new_lut_tvkeys=new_lut_4096
    )
    self._write_vac_to_tv(vac_write_json, on_finished=_after_write)