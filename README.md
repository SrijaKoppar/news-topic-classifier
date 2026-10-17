# News Topic Classifier

Classifies a news headline plus short description into one of nine sections and abstains when unsure. Trained only on `news.csv` (News Category Dataset, Misra, arXiv:2209.11429, CC BY 4.0; text © HuffPost). Built with Claude's assistance; I reviewed the code and can explain it.

## Data and design
- 25,000 rows; imbalanced (LIFESTYLE 24.0%, EDUCATION 2.0%), so the main metric is **macro-F1**.
- Inputs: headline + description only (what the API receives). Excluded `id`, `link`, `authors`, and `date` (class mix varies by year, a shortcut that fails on split_b).
- 9.3% of rows lack a description: kept (optional in the API). 12 repeated texts: left as is. Cleaning is whitespace-only.
- The two splits are separate experiments, never combined. For each, a stratified 15% of train is the validation set (seed 42), used for all tuning and the abstention threshold. Tests are scored once, after the model choice was committed; unit tests check no test row reaches train/validation.
- `split_a` is random. `split_b` is time-based (train 2012–16, test 2017–22) with a large class-mix shift (POLITICS 16.7% → 45.6%).

## Models and test results
Baseline: TF-IDF + logistic regression. Stronger: frozen `all-MiniLM-L6-v2` embeddings + logistic regression (CPU, not fine-tuned).

| Model | split_a macro-F1 / acc | split_b macro-F1 / acc |
|---|---:|---:|
| TF-IDF + LR | 0.685 / 0.755 | 0.560 / 0.711 |
| MiniLM + LR | **0.695 / 0.769** | **0.606 / 0.748** |

Embeddings win on both: +0.010 on split_a (within noise), +0.046 on split_b; no significance tests. Weakest: TECH & SCIENCE. Per-class tables and confusion matrices are in `results/`.
**Why splits differ:** validation-to-test macro-F1 fell 0.005–0.012 on split_a but 0.088–0.110 on split_b. The class-mix shift is the likely cause (untested: vocabulary/label drift). **I trust split_b's ≈0.61** for future articles; its small classes (62–80 rows) are noisy.

## Errors
30 sampled test errors (one annotator, stratified by confidence): 11 topic overlap, 7 WORLD/POLITICS boundary, 5 politics-heavy TV items labelled ENTERTAINMENT (all split_b), 3 questionable labels, 2 too little text, 2 clear model misses. Counts describe the sample, not frequencies. See `results/error_analysis_findings.md`.

## Abstention
Abstains (`category: null`) when top probability < **0.50**, chosen on validation only: best mean macro-F1 on answered cases with coverage ≥ 80% on both splits. **The 80% floor is my assumption** (70% → 0.60, 90% → 0.40). Test check, threshold unchanged: coverage 85% / 82%, macro-F1 on answered 0.746 / 0.687 vs 0.695 / 0.606 without abstention. It catches only 36–38% of errors; confidence is not a guarantee.

## Serving, latency, cost
Serves the embedding model fitted on all split_a train rows (assumption; not compared with a split_b model). Mean 6.0 ms/request (p50 5.8, p95 8.3; 300 sequential in-process requests, laptop CPU) ≈ 6 s per 1,000 predictions; $0 beyond local compute.

## One more week
Time-aware validation, bootstrap confidence intervals, per-class calibration, a larger encoder, multi-annotator label review.

## Run it
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Linux CPU-only: first pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
bash scripts/reproduce.sh         # needs internet once for the encoder
uvicorn src.api:app --port 8000
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"headline": "Senate passes new budget bill", "short_description": "Lawmakers voted late on Thursday."}'
```
`models/` is rebuilt by `reproduce.sh`, not committed. A fresh-clone run on my machine reproduced every metric (only fit timings differ); other machines untested.