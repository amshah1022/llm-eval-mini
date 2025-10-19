import pandas as pd
from pathlib import Path
from src.pipeline.run_eval import aggregate_per_prompt, reliability_alpha
from src.eval.metrics import wilcoxon_signed_rank, rank_biserial_from_wilcoxon

def main(control_path="data/ratings_control.csv", rubric_path="data/ratings_rubric.csv", outdir="reports"):
    out = Path(outdir); out.mkdir(exist_ok=True)
    ctrl = pd.read_csv(control_path)
    rubr = pd.read_csv(rubric_path)
    m_ctrl = aggregate_per_prompt(ctrl)
    m_rubr = aggregate_per_prompt(rubr)
    x = m_ctrl.loc[sorted(m_ctrl.index)].values.astype(float)
    y = m_rubr.loc[m_ctrl.index].values.astype(float)
    w = wilcoxon_signed_rank(x, y)
    r_rb = rank_biserial_from_wilcoxon(x, y)
    alpha_ctrl = reliability_alpha(ctrl)
    alpha_rubr = reliability_alpha(rubr)
    df = pd.DataFrame([{
        "n_prompts": len(x),
        "alpha_control": round(alpha_ctrl,3),
        "alpha_rubric": round(alpha_rubr,3),
        "wilcoxon_W": round(w["W"],3),
        "wilcoxon_Z": round(w["Z"],3),
        "p_two_sided": round(w["p_two_sided"],4),
        "effect_r": round(w["r"],3),
        "rank_biserial": round(r_rb,3)
    }])
    df.to_csv(out / "stress_summary.tsv", sep='\t', index=False)

if __name__ == "__main__":
    main()
