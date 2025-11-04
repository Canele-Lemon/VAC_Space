def _run_batch_correction_with_jacobian(
    self,
    iter_idx=1,
    max_iters=2,
    thr_gamma=0.05,
    thr_c=0.003,
    lam=1e-3,
    metrics=None,
):
    """
    OFF/ON 전체 측정 결과를 바탕으로:
      1) NG gray 리스트 추출 (0,1,254,255는 이미 제외된 상태라고 가정)
      2) 각 NG g에 대해 J_g로 ΔR_H,ΔG_H,ΔB_H 계산
      3) LUT index mapping을 이용해 High LUT의 해당 index에 Δ를 누적
      4) index별 평균 Δ 적용 → monotone 보장 → TV에 한 번에 write
      5) 전체 ON 재측정 → _on_spec_eval_done으로 다시 평가
    """
    import numpy as np
    import logging

    logging.info(f"[BATCH CORR] iteration {iter_idx} start (Jacobian dense)")

    # 0) 자코비안 / LUT / VAC 캐시 체크
    if not hasattr(self, "_J_dense"):
        logging.error("[BATCH CORR] J_dense not loaded (self._J_dense 없음)")
        return

    # LUT index mapping 로드 (gray→High LUT index)
    self._load_lut_mapping_high()

    if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
        logging.error("[BATCH CORR] no VAC cache; need latest TV VAC JSON")
        return

    # 1) NG gray / Δ타깃 준비
    if metrics is not None and "ng_grays" in metrics and "dG" in metrics:
        # 🔸 SpecEvalThread 결과 재사용
        ng_list = list(metrics["ng_grays"])
        d_targets = {
            "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
            "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
            "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
        }
        thr_gamma = float(metrics.get("thr_gamma", thr_gamma))
        thr_c     = float(metrics.get("thr_c",     thr_c))
        logging.info(f"[BATCH CORR] reuse metrics from SpecEvalThread, NG grays={ng_list}")
    else:
        # 🔸 폴백: SpecEvalThread helper를 직접 사용해서 다시 계산
        from .SpecEvalThread import SpecEvalThread  # 경로는 실제 모듈 구조에 맞게 조정
        dG, dCx, dCy, ng_list = SpecEvalThread.compute_gray_errors_and_ng_list(
            self._off_store, self._on_store,
            thr_gamma=thr_gamma, thr_c=thr_c
        )
        d_targets = {
            "Gamma": dG.astype(np.float32),
            "Cx":    dCx.astype(np.float32),
            "Cy":    dCy.astype(np.float32),
        }
        logging.info(f"[BATCH CORR] NG grays (recomputed): {ng_list}")

    if not ng_list:
        logging.info("[BATCH CORR] no NG gray (또는 0/1/254/255만 NG) → 보정 없음")
        return

    # 2) TV에서 현재 High LUT 확보
    vac_dict = self._vac_dict_cache

    RH0 = np.asarray(vac_dict["RchannelHigh"], dtype=np.float32).copy()
    GH0 = np.asarray(vac_dict["GchannelHigh"], dtype=np.float32).copy()
    BH0 = np.asarray(vac_dict["BchannelHigh"], dtype=np.float32).copy()

    RH = RH0.copy()
    GH = GH0.copy()
    BH = BH0.copy()

    # 3) index별 Δ 누적 (여러 gray가 같은 index를 참조할 수 있으므로)
    delta_acc = {
        "R": np.zeros_like(RH),
        "G": np.zeros_like(GH),
        "B": np.zeros_like(BH),
    }
    count_acc = {
        "R": np.zeros_like(RH, dtype=np.int32),
        "G": np.zeros_like(GH, dtype=np.int32),
        "B": np.zeros_like(BH, dtype=np.int32),
    }

    mapR = self._lut_map_high["R"]   # (256,)
    mapG = self._lut_map_high["G"]
    mapB = self._lut_map_high["B"]

    # 4) 각 NG gray에 대해 ΔR/G/B 계산 후 index에 누적
    for g in ng_list:
        dX = self._solve_delta_rgb_for_gray(
            g,
            d_targets,
            lam=lam,
            wCx=0.5,
            wCy=0.5,
            wG=1.0,
        )
        if dX is None:
            continue

        dR, dG, dB = dX

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

    # 5) index별 평균 Δ 적용 + clip + monotone + 로그
    for ch, arr, arr0 in (
        ("R", RH, RH0),
        ("G", GH, GH0),
        ("B", BH, BH0),
    ):
        da = delta_acc[ch]
        ct = count_acc[ch]
        mask = ct > 0

        if not np.any(mask):
            logging.info(f"[BATCH CORR] channel {ch}: no indices updated")
            continue

        # 평균 Δ
        arr[mask] = arr0[mask] + (da[mask] / ct[mask])
        # clip
        arr[:] = np.clip(arr, 0.0, 4095.0)
        # 단조 증가 (i<j → LUT[i] ≤ LUT[j])
        self._enforce_monotone(arr)

        # 🔹 인덱스별 보정 로그 (before → after)
        changed_idx = np.where(mask)[0]
        logging.info(f"[BATCH CORR] channel {ch}: {len(changed_idx)} indices updated")
        for idx in changed_idx:
            before = float(arr0[idx])
            after  = float(arr[idx])
            delta  = after - before
            logging.debug(
                f"[BATCH CORR] ch={ch} idx={idx:4d}: {before:7.1f} → {after:7.1f} (Δ={delta:+.2f})"
            )

    # 6) NG gray 기준으로 어떤 LUT index가 어떻게 바뀌었는지 추가 요약 로그
    for g in ng_list:
        idxR = int(mapR[g])
        idxG = int(mapG[g])
        idxB = int(mapB[g])
        info = []
        if 0 <= idxR < len(RH0):
            info.append(
                f"R(idx={idxR}): {RH0[idxR]:.1f}→{RH[idxR]:.1f} (Δ={RH[idxR]-RH0[idxR]:+.1f})"
            )
        if 0 <= idxG < len(GH0):
            info.append(
                f"G(idx={idxG}): {GH0[idxG]:.1f}→{GH[idxG]:.1f} (Δ={GH[idxG]-GH0[idxG]:+.1f})"
            )
        if 0 <= idxB < len(BH0):
            info.append(
                f"B(idx={idxB}): {BH0[idxB]:.1f}→{BH[idxB]:.1f} (Δ={BH[idxB]-BH0[idxB]:+.1f})"
            )
        if info:
            logging.info(f"[BATCH CORR] g={g:3d} → " + " | ".join(info))

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
        new_lut_4096[k] = np.clip(np.round(new_lut_4096[k]), 0, 4095).astype(np.uint16)

    # UI용 플롯 dict
    lut_dict_plot = {
        "R_Low":  new_lut_4096["RchannelLow"],
        "R_High": new_lut_4096["RchannelHigh"],
        "G_Low":  new_lut_4096["GchannelLow"],
        "G_High": new_lut_4096["GchannelHigh"],
        "B_Low":  new_lut_4096["BchannelLow"],
        "B_High": new_lut_4096["BchannelHigh"],
    }
    self._update_lut_chart_and_table(lut_dict_plot)

    # 8) TV write → read → 전체 ON 재측정 → Spec 재평가
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
                lambda ok, m: self._on_spec_eval_done(ok, m, iter_idx, max_iters)
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