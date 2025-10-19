"""
Regenerate synthetic ratings with a controlled advantage for the rubric condition.
"""
import random
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def main(seed=7, n_prompts=12, n_raters=4, bias_rubric=0.4):
    random.seed(seed)
    prompts = pd.read_csv(DATA / "prompts.csv")
    raters = [f"R{i+1}" for i in range(n_raters)]

    def gen(bias=0.0):
        rows = []
        for p in prompts["prompt_id"]:
            base_mu = random.uniform(3.6, 5.4)
            for r in raters:
                val = base_mu + bias + random.gauss(0, 0.7)
                val = int(min(7, max(1, round(val))))
                rows.append({"prompt_id": p, "rater_id": r, "rating": val})
        return pd.DataFrame(rows)

    ctrl = gen(0.0)
    rubr = gen(bias_rubric)

    ctrl.to_csv(DATA / "ratings_control.csv", index=False)
    rubr.to_csv(DATA / "ratings_rubric.csv", index=False)
    print("Regenerated:", DATA / "ratings_control.csv")
    print("Regenerated:", DATA / "ratings_rubric.csv")

if __name__ == "__main__":
    main()
