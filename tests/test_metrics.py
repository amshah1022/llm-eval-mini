import numpy as np
from src.eval.metrics import wilcoxon_signed_rank, rank_biserial_from_wilcoxon, krippendorff_alpha_interval

def test_wilcoxon_basic():
    x = [1, 2, 3, 4]
    y = [2, 3, 4, 5]
    res = wilcoxon_signed_rank(x, y)
    assert res["n_effective"] == 4
    assert res["p_two_sided"] <= 0.2

def test_rank_biserial_range():
    x = [1, 2, 3]
    y = [3, 3, 3]
    rrb = rank_biserial_from_wilcoxon(x, y)
    assert -1.0 <= rrb <= 1.0

def test_krippendorff_alpha_extremes():
    M = np.array([[1,1,1],[2,2,2]], dtype=float)
    a = krippendorff_alpha_interval(M)
    assert a > 0.95
