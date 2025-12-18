@dataclass(frozen=True)
class VACSpecPolicy:
    thr_gamma: float = 0.05
    thr_c: float = 0.003

    gamma_eval_grays: FrozenSet[int] = frozenset(range(2, 248))
    color_eval_grays: FrozenSet[int] = frozenset(range(6, 256))

    gamma_style_excluded: FrozenSet[int] = frozenset({0, 1, *range(248, 256)})
    color_style_excluded: FrozenSet[int] = frozenset(range(0, 6))

    def color_ok(self, d_cx: float, d_cy: float) -> bool:
        return (abs(d_cx) <= self.thr_c) and (abs(d_cy) <= self.thr_c)

    def gamma_ok(self, d_g: float) -> bool:
        return abs(d_g) <= self.thr_gamma

    def should_style_color(self, gray: int) -> bool:
        return gray in self.color_eval_grays and gray not in self.color_style_excluded

    def should_style_gamma(self, gray: int) -> bool:
        return gray in self.gamma_eval_grays and gray not in self.gamma_style_excluded