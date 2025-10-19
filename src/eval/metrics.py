# Minimal evaluation metrics: paired Wilcoxon (with normal approximation), effect sizes,
# and Krippendorff's alpha for interval data. No external dependencies beyond pandas/numpy/matplotlib.

from math import erf, sqrt
from typing import List, Tuple
import numpy as np

def _normal_cdf(x: float) -> float:
    # Standard normal CDF using erf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def wilcoxon_signed_rank(x: List[float], y: List[float]) -> dict:
    """
    Paired Wilcoxon signed-rank test (two-sided) with large-sample normal approximation.
    Returns dict with W, Z, p_two_sided, n_effective, r (Z/sqrt(n)).
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    diffs = np.array(y, dtype=float) - np.array(x, dtype=float)
    # Remove zeros
    nonzero = diffs != 0
    diffs = diffs[nonzero]
    n = len(diffs)
    if n == 0:
        return {"W": 0.0, "Z": 0.0, "p_two_sided": 1.0, "n_effective": 0, "r": 0.0}
    abs_d = np.abs(diffs)
    ranks = rankdata(abs_d)  # average ranks for ties
    signed_ranks = ranks * np.sign(diffs)
    W_pos = signed_ranks[signed_ranks > 0].sum()
    W_neg = -signed_ranks[signed_ranks < 0].sum()
    W = min(W_pos, W_neg)  # conventional statistic

    # Normal approximation with tie correction
    T = tie_correction(abs_d)
    mu_W = n * (n + 1) / 4.0
    sigma_W = sqrt((n * (n + 1) * (2*n + 1) / 24.0) * T)
    if sigma_W == 0:
        Z = 0.0
        p = 1.0
    else:
        # Continuity correction
        Z = (W - mu_W - 0.5) / sigma_W
        p = 2 * (1 - _normal_cdf(abs(Z)))
    r = Z / sqrt(n) if n > 0 else 0.0
    return {"W": float(W), "Z": float(Z), "p_two_sided": float(p), "n_effective": int(n), "r": float(r)}

def rank_biserial_from_wilcoxon(x: List[float], y: List[float]) -> float:
    """
    Rank-biserial correlation derived from Wilcoxon signed ranks.
    r_rb = (R_plus - R_minus) / (n*(n+1)/2)
    """
    diffs = np.array(y, dtype=float) - np.array(x, dtype=float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0
    abs_d = np.abs(diffs)
    ranks = rankdata(abs_d)
    signed_ranks = ranks * np.sign(diffs)
    R_plus = signed_ranks[signed_ranks > 0].sum()
    R_minus = -signed_ranks[signed_ranks < 0].sum()
    denom = n * (n + 1) / 2.0
    return float((R_plus - R_minus) / denom) if denom > 0 else 0.0

def tie_correction(values: np.ndarray) -> float:
    """
    Tie correction factor for Wilcoxon variance (1 - sum(t_i^3 - t_i)/(n^3 - n))
    where t_i are tie group sizes. For no ties, returns 1.0.
    """
    _, counts = np.unique(values, return_counts=True)
    n = values.size
    if n < 2:
        return 1.0
    tie_sum = np.sum(counts**3 - counts)
    return 1.0 - tie_sum / (n**3 - n) if (n**3 - n) != 0 else 1.0

def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Average rank for ties (1-based). Minimal replacement for scipy.stats.rankdata.
    """
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a)+1, dtype=float)
    # handle ties
    _, inv, counts = np.unique(a[order], return_inverse=True, return_counts=True)
    cumsum = np.cumsum(counts)
    starts = cumsum - counts + 1
    avg = (starts + cumsum) / 2.0
    ranks[order] = avg[inv]
    return ranks

def krippendorff_alpha_interval(data: np.ndarray) -> float:
    """
    Krippendorff's alpha for interval data.
    Data shape: items x raters, with np.nan for missing.
    """
    # mask missing
    mask = ~np.isnan(data)
    # overall mean (weighted by available)
    values = data[mask]
    if values.size == 0:
        return np.nan
    mean_all = np.nanmean(data)
    # observed disagreement Do
    Do_num, Do_den = 0.0, 0.0
    for i in range(data.shape[0]):
        row = data[i, :]
        m = ~np.isnan(row)
        n_i = m.sum()
        if n_i > 1:
            diffs = row[m][:, None] - row[m][None, :]
            Do_num += np.nansum(diffs**2)
            Do_den += n_i * (n_i - 1)
    Do = Do_num / Do_den if Do_den > 0 else 0.0

    # expected disagreement De
    diffs_all = values[:, None] - values[None, :]
    De = np.nansum(diffs_all**2) / (values.size * (values.size - 1)) if values.size > 1 else 0.0

    if De == 0:
        return 1.0 if Do == 0 else np.nan
    return 1.0 - Do / De
