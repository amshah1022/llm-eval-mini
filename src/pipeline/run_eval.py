import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.eval.metrics import wilcoxon_signed_rank, rank_biserial_from_wilcoxon, krippendorff_alpha_interval

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

def load():
    prompts = pd.read_csv(DATA / "prompts.csv")
    ctrl = pd.read_csv(DATA / "ratings_control.csv")
    rubr = pd.read_csv(DATA / "ratings_rubric.csv")
    return prompts, ctrl, rubr

def aggregate_per_prompt(ratings: pd.DataFrame) -> pd.Series:
    # median across raters per prompt (Likert)
    return ratings.groupby("prompt_id")["rating"].median().sort_index()

def reliability_alpha(ratings: pd.DataFrame) -> float:
    # items x raters matrix with NaN for missing
    items = sorted(ratings["prompt_id"].unique())
    raters = sorted(ratings["rater_id"].unique())
    M = np.full((len(items), len(raters)), np.nan, dtype=float)
    i_map = {p:i for i,p in enumerate(items)}
    r_map = {r:i for i,r in enumerate(raters)}
    for _, row in ratings.iterrows():
        M[i_map[row["prompt_id"]], r_map[row["rater_id"]]] = float(row["rating"])
    return float(krippendorff_alpha_interval(M))

def run():
    prompts, ctrl, rubr = load()

    # reliability per condition
    alpha_ctrl = reliability_alpha(ctrl)
    alpha_rubr = reliability_alpha(rubr)

    # aggregate medians per prompt
    m_ctrl = aggregate_per_prompt(ctrl)
    m_rubr = aggregate_per_prompt(rubr)

    # align order
    m_ctrl = m_ctrl.loc[sorted(m_ctrl.index)]
    m_rubr = m_rubr.loc[m_ctrl.index]

    # paired stats
    x = m_ctrl.values.astype(float)
    y = m_rubr.values.astype(float)
    w = wilcoxon_signed_rank(x, y)
    r_rb = rank_biserial_from_wilcoxon(x, y)

    # Save summary
    summary = pd.DataFrame([{
        "n_prompts": len(x),
        "alpha_control": round(alpha_ctrl, 3),
        "alpha_rubric": round(alpha_rubr, 3),
        "wilcoxon_W": round(w["W"], 3),
        "wilcoxon_Z": round(w["Z"], 3),
        "p_two_sided": round(w["p_two_sided"], 4),
        "effect_r": round(w["r"], 3),
        "rank_biserial": round(r_rb, 3),
        "median_control": float(np.median(x)),
        "median_rubric": float(np.median(y)),
        "median_delta": float(np.median(y - x)),
    }])
    summary.to_csv(REPORTS / "summary.tsv", sep="\t", index=False)

    # Plot paired deltas
    deltas = y - x
    fig = plt.figure(figsize=(6,4))
    plt.axhline(0, linewidth=1)
    plt.scatter(range(1, len(deltas)+1), deltas)
    plt.xlabel("Prompt index")
    plt.ylabel("Rubric − Control (median rating)")
    plt.title("Per-prompt paired differences")
    plt.tight_layout()
    figpath = REPORTS / "delta_plot.png"
    plt.savefig(figpath, dpi=150)
    print("Saved:", figpath)
    print("Saved:", REPORTS / "summary.tsv")

if __name__ == "__main__":
    run()
