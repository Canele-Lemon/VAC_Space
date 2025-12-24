thr_gamma와 the_c는 아래 클래스에서 관리하고 있습니다
from dataclasses import dataclass
from typing import FrozenSet

@dataclass(frozen=True)
class VACSpecPolicy:
    thr_gamma: float = 0.05
    thr_c: float = 0.003

    gamma_eval_grays: FrozenSet[int] = frozenset(range(2, 248))
    color_eval_grays: FrozenSet[int] = frozenset(range(6, 256))
    
    def should_eval_gamma(self, gray: int) -> bool:
        return gray in self.gamma_eval_grays

    def should_eval_color(self, gray: int) -> bool:
        return gray in self.color_eval_grays

    def gamma_ok(self, d_g: float) -> bool:
        return abs(d_g) <= self.thr_gamma

    def color_ok(self, d_cx: float, d_cy: float) -> bool:
        return (abs(d_cx) <= self.thr_c) and (abs(d_cy) <= self.thr_c)

이걸 써서 _on_spec_eval_done포함 이어지는 보정 메서드를 전체적으로 수정해주세요. 아래는 현재까지 수정한 코드입니다. (코드 전체 보여주세요)
    def _on_spec_eval_done(self, spec_ok, metrics, iter_idx, max_iters):
        """
        조건 1) spec_ok==True: 종료
        조건 2) (spec_ok==False) and (max_iters>0): NG Gray Correction
        """
        try:
            ng_grays = []
            thr_g = None
            thr_c = None
            
            if metrics and "error" not in metrics:
                max_dG   = metrics.get("max_dG",  float("nan"))
                max_dCx  = metrics.get("max_dCx", float("nan"))
                max_dCy  = metrics.get("max_dCy", float("nan"))
                thr_g    = metrics.get("thr_gamma", self._spec_thread.thr_gamma if self._spec_thread else None)
                thr_c    = metrics.get("thr_c",     self._spec_thread.thr_c     if self._spec_thread else None)
                ng_grays = metrics.get("ng_grays", [])
                
                logging.info(
                    f"[Evaluation] max|ΔGamma|={max_dG:.6f} (≤{thr_g}), "
                    f"max|ΔCx|={max_dCx:.6f}, max|ΔCy|={max_dCy:.6f} (≤{thr_c}), "
                    f"NG grays={ng_grays}"
                )
            else:
                logging.warning("[Evaluation] evaluation failed — treating as not passed.")
                ng_grays = []

            self._update_spec_views(iter_idx, self._off_store, self._on_store)

            # 조건 1) spec_ok==True: 종료
            if spec_ok:
                self._step_done(5)
                logging.info("[Evaluation] Spec 통과 — 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            # 조건 2) (spec_ok==False) and (max_iters>0): NG Gray Correction
            self._step_fail(5)
            
            if max_iters <= 0:
                logging.info("[Evaluation] Spec NG but no further correction (max_iters≤0) - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            if iter_idx >= max_iters:
                logging.info("[Evaluation] Spec NG but 보정 횟수 초과 - 최적화 종료")
                self.ui.vac_btn_JSONdownload.setEnabled(True)
                return
            
            for s in (2, 3, 4):
                self._step_set_pending(s)
                
            thr_gamma = float(thr_g) if thr_g is not None else 0.05
            thr_c_val = float(thr_c) if thr_c is not None else 0.003

            self._run_batch_correction_with_jacobian(
                iter_idx=iter_idx+1,
                max_iters=max_iters,
                thr_gamma=thr_gamma,
                thr_c=thr_c_val,
                metrics=metrics
            )
            return
            
            # 무시하세요
            # if not getattr(self, "_failover_vac_applied3335", False):
            #         logging.info("[Failover] Spec NG — 대체 VAC(pk=3335) 적용 및 재평가 시도")
            #         self._failover_vac_applied3335 = True
            #         thr_gamma = float(thr_g) if thr_g is not None else 0.05
            #         thr_c_val = float(thr_c) if thr_c is not None else 0.003
            #         self._apply_vac_by_pk_and_re_evaluate(
            #             vac_info_pk=3336,
            #             thr_gamma=thr_gamma,
            #             thr_c=thr_c_val
            #         )
            #         return
            
        finally:
            self._spec_thread = None


