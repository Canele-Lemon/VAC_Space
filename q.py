def _run_batch_correction_with_jacobian(self, iter_idx=1, max_iters=2,
                                        thr_gamma=0.05, thr_c=0.003,
                                        lam=1e-3):
    """
    OFF/ON 전체 측정 결과를 바탕으로:
      1) NG gray 리스트 추출 (0,1,254,255 제외)
      2) 각 NG g에 대해 J_g로 ΔR_H,ΔG_H,ΔB_H 계산
      3) mapping CSV를 이용해 High LUT의 해당 index에 누적
      4) 모든 채널에 대해 monotone enforcement 후 TV에 한 번에 write
      5) 전체 ON 재측정 → spec 평가(_on_spec_eval_done에 다시 들어감)
    """

    logging.info(f"[BATCH CORR] iteration {iter_idx} start (Jacobian dense)")

    # 0) 사전 조건: 자코비안 & LUT mapping & VAC cache
    if not hasattr(self, "_J_dense"):
        logging.error("[BATCH CORR] J_dense not loaded")
        return
    self._load_lut_mapping_high()
    if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
        logging.error("[BATCH CORR] no VAC cache; need latest TV VAC JSON")
        return

    # 1) Δ 타깃과 NG gray 리스트
    ng_list, d_targets = self._get_ng_gray_list(
        self._off_store, self._on_store,
        thr_gamma=thr_gamma, thr_c=thr_c
    )
    logging.info(f"[BATCH CORR] NG grays: {ng_list}")

    if not ng_list:
        logging.info("[BATCH CORR] no NG gray (or only edge NG) → nothing to correct")
        return

    vac_dict = self._vac_dict_cache
    # 2) High LUT 4096 배열 준비
    RH = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
    GH = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
    BH = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

    # 3) index별 Δ 누적 (여러 gray가 같은 index를 참조할 수 있으므로)
    delta_acc = { "R": np.zeros_like(RH), "G": np.zeros_like(GH), "B": np.zeros_like(BH) }
    count_acc = { "R": np.zeros_like(RH, dtype=np.int32),
                  "G": np.zeros_like(GH, dtype=np.int32),
                  "B": np.zeros_like(BH, dtype=np.int32) }

    mapR = self._lut_map_high["R"]   # (256,)
    mapG = self._lut_map_high["G"]
    mapB = self._lut_map_high["B"]

    for g in ng_list:
        dX = self._solve_delta_rgb_for_gray(g, d_targets, lam=lam,
                                            wCx=0.5, wCy=0.5, wG=1.0)
        if dX is None:
            continue

        dR, dG, dB = dX  # ΔR_H,ΔG_H,ΔB_H (12bit count 단위)

        idxR = int(mapR[g])
        idxG = int(mapG[g])
        idxB = int(mapB[g])

        if 0 <= idxR < len(RH):
            delta_acc["R"][idxR] += dR
            count_acc["R"][idxR] += 1
        if 0 <= idxG < len(GH):
            delta_acc["G"][idxG] += dG
            count_acc["G"][idxG] += 1
        if 0 <= idxB < len(BH):
            delta_acc["B"][idxB] += dB
            count_acc["B"][idxB] += 1

    # 4) 평균 Δ 적용
    for ch, arr in (("R", RH), ("G", GH), ("B", BH)):
        da = delta_acc[ch]; ct = count_acc[ch]
        mask = ct > 0
        arr[mask] += (da[mask] / ct[mask])
        # clip 먼저
        arr[:] = np.clip(arr, 0.0, 4095.0)
        # 단조 증가 보장
        self._enforce_monotone(arr)

        if ch == "R":
            RH = arr
        elif ch == "G":
            GH = arr
        else:
            BH = arr

    # 5) 새 4096 LUT 구성 (Low는 그대로, High만 업데이트)
    new_lut_4096 = {
        "RchannelLow":  np.asarray(vac_dict["RchannelLow"],  dtype=np.float32),
        "GchannelLow":  np.asarray(vac_dict["GchannelLow"],  dtype=np.float32),
        "BchannelLow":  np.asarray(vac_dict["BchannelLow"],  dtype=np.float32),
        "RchannelHigh": RH,
        "GchannelHigh": GH,
        "BchannelHigh": BH,
    }
    for k in new_lut_4096:
        new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

    # UI용 plot dict
    lut_dict_plot = {
        "R_Low":  new_lut_4096["RchannelLow"],
        "R_High": new_lut_4096["RchannelHigh"],
        "G_Low":  new_lut_4096["GchannelLow"],
        "G_High": new_lut_4096["GchannelHigh"],
        "B_Low":  new_lut_4096["BchannelLow"],
        "B_High": new_lut_4096["BchannelHigh"],
    }
    self._update_lut_chart_and_table(lut_dict_plot)

    # 6) TV write → read → 전체 ON 재측정 (기존 _run_correction_iteration 흐름 재사용)
    logging.info(f"[BATCH CORR] LUT apply iter={iter_idx}")

    vac_write_json = self.build_vacparam_std_format(
        base_vac_dict=self._vac_dict_cache,
        new_lut_tvkeys=new_lut_4096
    )

    def _after_write(ok, msg):
        logging.info(f"[BATCH CORR] write result: {ok} {msg}")
        if not ok:
            return
        logging.info("[BATCH CORR] TV reading after write")
        self._read_vac_from_tv(_after_read_back)

    def _after_read_back(vac_dict_after):
        if not vac_dict_after:
            logging.error("[BATCH CORR] TV read-back failed")
            return
        self._vac_dict_cache = vac_dict_after
        self._step_done(3)

        # ON 시리즈 리셋
        self.vac_optimization_gamma_chart.reset_on()
        self.vac_optimization_cie1976_chart.reset_on()

        profile_corr = SessionProfile(
            legend_text=f"CORR #{iter_idx}",
            cie_label=None,
            table_cols={"lv":4, "cx":5, "cy":6, "gamma":7,
                        "d_cx":8, "d_cy":9, "d_gamma":10},
            ref_store=self._off_store
        )

        def _after_corr(store_corr):
            self._step_done(4)
            self._on_store = store_corr
            self._step_start(5)
            self._spec_thread = SpecEvalThread(
                self._off_store, self._on_store,
                thr_gamma=thr_gamma, thr_c=thr_c, parent=self
            )
            self._spec_thread.finished.connect(
                lambda ok, metrics: self._on_spec_eval_done(ok, metrics, iter_idx, max_iters)
            )
            self._spec_thread.start()

        logging.info("[BATCH CORR] re-measure start (after LUT update)")
        self._step_start(4)
        self.start_viewing_angle_session(
            profile=profile_corr,
            gray_levels=op.gray_levels_256,
            gamma_patterns=('white',),
            colorshift_patterns=op.colorshift_patterns,
            first_gray_delay_ms=3000, cs_settle_ms=1000,
            on_done=_after_corr
        )

    self._step_start(3)
    self._write_vac_to_tv(vac_write_json, on_finished=_after_write)
    
def _solve_delta_rgb_for_gray(self, g, d_targets, lam=1e-3,
                              wCx=0.5, wCy=0.5, wG=1.0):
    """
    하나의 gray g에 대해,
      ΔY = [ΔCx, ΔCy, ΔGamma] (ON-OFF) 가 주어졌을 때
    'OFF와 같게' 만들기 위한 ΔX = [ΔR_H,ΔG_H,ΔB_H] 를 구한다.
    
    ΔY_target = -ΔY 를 사용.
    자코비안: J_g (3×3)  rows=[Cx,Cy,Gamma], cols=[R_H,G_H,B_H]
    """

    Jg = self._J_dense[g]           # (3,3)
    if not np.isfinite(Jg).all():
        return None  # 이 gray는 자코비안 신뢰 불가

    # 샘플 수/condition도 체크 (옵션)
    if self._J_n[g] < 3:
        return None
    if not np.isfinite(self._J_cond[g]) or self._J_cond[g] > 1e6:
        # 너무 ill-conditioned 이면 skip
        return None

    dCx = float(d_targets["Cx"][g])
    dCy = float(d_targets["Cy"][g])
    dG  = float(d_targets["Gamma"][g])

    # 이미 거의 맞은 gray는 굳이 보정 안함 (추가 데드밴드)
    if (abs(dCx) < 1e-4) and (abs(dCy) < 1e-4) and (abs(dG) < 1e-3):
        return None

    # OFF와 같게 만들려면 ΔY_target = -(ΔY_measured)
    Y = np.array([-dCx, -dCy, -dG], dtype=np.float64)

    # 색/감마 중요도 가중치
    W = np.diag([wCx, wCy, wG])     # (3,3)
    Jw = W @ Jg                     # (3,3)
    Yw = W @ Y                      # (3,)

    # 리지 최소자승 ΔX = (Jwᵀ Jw + λI)^{-1} Jwᵀ Yw
    JTJ = Jw.T @ Jw                 # (3,3)
    JTY = Jw.T @ Yw                # (3,)

    JTJ = JTJ + lam * np.eye(3, dtype=np.float64)
    try:
        dX = np.linalg.solve(JTJ, JTY)   # (3,)
    except np.linalg.LinAlgError:
        dX = np.linalg.pinv(JTJ) @ JTY

    return dX.astype(np.float32)   # [ΔR_H,ΔG_H,ΔB_H]
    
def _load_lut_mapping_high(self):
    """
    실행 py 파일 폴더에 있는 LUT_index_mapping.csv 를 읽어
    각 gray별 High LUT index를 저장.
    
    CSV 예시 가정:
        gray,R_High,G_High,B_High
        0,0,0,0
        1,16,16,16
        ...
    """
    if hasattr(self, "_lut_map_high") and self._lut_map_high is not None:
        return  # 이미 로드됨

    csv_path = cf.get_normalized_path(__file__, '.', 'LUT_index_mapping.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"LUT_index_mapping.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 🔧 컬럼명은 실제 파일에 맞게 조정 필요
    self._lut_map_high = {
        "R": df["R_High"].to_numpy(dtype=np.int32),
        "G": df["G_High"].to_numpy(dtype=np.int32),
        "B": df["B_High"].to_numpy(dtype=np.int32),
    }
    logging.info(f"[LUT MAP] loaded {csv_path}, shape={df.shape}")