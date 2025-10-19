# LLM Evaluation Mini — Reliability & Oversight

A tiny, reproducible evaluation suite that compares two LLM systems on rubric-based feedback ratings.
It demonstrates **paired analysis** (Wilcoxon signed-rank, rank-biserial effect size) and **inter-rater reliability**
(Krippendorff's alpha, interval) with a simple plot and a clean report — ideal as a portfolio artifact
for roles focused on **evaluation infrastructure**, **data quality**, and **oversight**.

> One weekend build: run `python -m src.pipeline.run_eval` and open `reports/summary.tsv` and `reports/delta_plot.png`.

## Why this exists
Preference ≠ reliability. This repo shows how to evaluate *feedback quality* using rubric-aligned ratings,
with transparent stats and a single reproducible script.

## Repo layout
```
llm-eval-mini/
├─ data/
│  ├─ prompts.csv                 # toy prompt IDs + text
│  ├─ ratings_control.csv         # mock 7-point ratings per prompt by rater for the control chatbot
│  └─ ratings_rubric.csv          # mock 7-point ratings per prompt by rater for the rubric-anchored chatbot
├─ src/
│  ├─ eval/metrics.py             # Wilcoxon (paired), rank-biserial, Krippendorff's alpha (interval)
│  └─ pipeline/run_eval.py        # loads data, aggregates, runs stats, saves plot + report
├─ notebooks/
│  └─ quickstart.ipynb            # walk-through (load → aggregate → test → plot)
├─ scripts/
│  └─ generate_mock_outputs.py    # re-generate synthetic ratings with a modest rubric advantage
├─ reports/                        # created on first run
├─ LICENSE
├─ requirements.txt
└─ README.md
```

## Quickstart
1) Create a virtual environment (recommended) and install:
```bash
pip install -r requirements.txt
```
2) Run the evaluation:
```bash
python -m src.pipeline.run_eval
```
3) Open the outputs:
- `reports/summary.tsv`
- `reports/delta_plot.png`

## Methods (brief)
- **Paired design:** Each prompt is rated under both conditions → paired Wilcoxon signed-rank on per-prompt medians.
- **Effect size:** Rank-biserial correlation `r_rb` and `r = Z / sqrt(N)`.
- **Reliability:** Krippendorff's alpha (interval) across raters per condition.

### Example Results (Synthetic Data)

Running the demo with mock ratings produces reproducible outputs:

| Metric | Value | Interpretation |
|:--|:--:|:--|
| n_prompts | 12 | paired comparisons |
| Krippendorff’s α (control / rubric) | 0.486 / 0.098 | moderate vs. low inter-rater reliability |
| Wilcoxon Z | –0.08 | near zero difference |
| p (two-sided) | 0.94 | not significant (expected for synthetic data) |
| Median Δ | 0.0 | identical medians |

![Paired differences plot](reports/delta_plot.png)

These results confirm the pipeline runs end-to-end and outputs reliability metrics and visualizations.  
When replaced with real ratings, the same analysis quantifies rubric anchoring effects on clarity, actionability, and reliability.


## Notes
- This repo uses **synthetic example data**; replace with your own ratings to run real analysis.
- No external APIs required.
- Charts use Matplotlib with default styles and single plots as required.
