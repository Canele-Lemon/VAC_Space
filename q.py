def _run_batch_correction_with_jacobian(
    self,
    iter_idx=1,
    max_iters=2,
    thr_gamma=0.05,
    thr_c=0.003,
    lam=1e-3,
    metrics=None,
):
    logging.info(f"[BATCH CORR] iteration {iter_idx} start (Jacobian dense)")

    # 0) 자코비안 / LUT / VAC 캐시 체크
    if not hasattr(self, "_J_dense"):
        logging.error("[BATCH CORR] J_dense not loaded")
        return
    self._load_lut_mapping_high()
    if not hasattr(self, "_vac_dict_cache") or self._vac_dict_cache is None:
        logging.error("[BATCH CORR] no VAC cache; need latest TV VAC JSON")
        return

    # 1) NG gray / Δ타깃 준비
    if metrics is not None and "ng_grays" in metrics and "dG" in metrics:
        # 🔸 SpecEvalThread에서 계산한 값을 그대로 재사용
        ng_list = list(metrics["ng_grays"])
        d_targets = {
            "Gamma": np.asarray(metrics["dG"],  dtype=np.float32),
            "Cx":    np.asarray(metrics["dCx"], dtype=np.float32),
            "Cy":    np.asarray(metrics["dCy"], dtype=np.float32),
        }
        # threshold도 metrics에 있으면 맞춰줌
        thr_gamma = float(metrics.get("thr_gamma", thr_gamma))
        thr_c     = float(metrics.get("thr_c",     thr_c))
        logging.info(f"[BATCH CORR] reuse metrics from SpecEvalThread, NG={ng_list}")
    else:
        # 🔸 폴백: 직접 다시 계산 (compute_gray_errors_and_ng_list 재사용하면 좋음)
        ng_list, d_targets = self._get_ng_gray_list(
            self._off_store, self._on_store,
            thr_gamma=thr_gamma, thr_c=thr_c
        )
        logging.info(f"[BATCH CORR] NG grays (recomputed): {ng_list}")

    if not ng_list:
        logging.info("[BATCH CORR] no NG gray (or only edge NG) → nothing to correct")
        return

    ...
    # 아래 나머지 보정 로직은 그대로 유지