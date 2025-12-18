from dataclasses import dataclass
from typing import FrozenSet

@dataclass(frozen=True)
class VACSpecPolicy:
    thr_gamma: float = 0.05
    thr_c: float = 0.003

    # ✅ "평가할 gray" 정의 (이 집합에 포함된 경우만 OK/NG 색칠)
    gamma_eval_grays: FrozenSet[int] = frozenset(range(2, 248))   # 2..247
    color_eval_grays: FrozenSet[int] = frozenset(range(6, 256))   # 6..255

    # ---- 평가 여부 ----
    def should_eval_gamma(self, gray: int) -> bool:
        return gray in self.gamma_eval_grays

    def should_eval_color(self, gray: int) -> bool:
        return gray in self.color_eval_grays

    # ---- 스펙 판정 ----
    def gamma_ok(self, d_g: float) -> bool:
        return abs(d_g) <= self.thr_gamma

    def color_ok(self, d_cx: float, d_cy: float) -> bool:
        return (abs(d_cx) <= self.thr_c) and (abs(d_cy) <= self.thr_c)