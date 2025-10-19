import numpy as np
from src.eval.metrics import (
    wilcoxon_signed_rank,
    rank_biserial_from_wilcoxon,
    krippendorff_alpha_interval,
)

def test_wilcoxon_basic():
    # Mixed-sign, non-degenerate diffs to avoid edge cases
    x = [1, 2, 3, 4, 5]
    y = [2, 1, 5, 4, 6]  # diffs: [+1, -1, +2, 0, +1] -> drops the 0 internally
    res = wilcoxon_signed_rank(x, y)
    assert res["n_effective"] >= 4
    assert 0 < res["p_two_sided"] <= 1.0
    assert -1.0 <= res["r"] <= 1.0

def test_rank_biserial_range():
    x = [1, 2, 3]
    y = [3, 3, 3]
    rrb = rank_biserial_from_wilcoxon(x, y)
    assert -1.0 <= rrb <= 1.0

def test_krippendorff_alpha_extremes():
    # perfect agreement -> alpha ~ 1
    M = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)
    a = krippendorff_alpha_interval(M)
    assert a > 0.95
