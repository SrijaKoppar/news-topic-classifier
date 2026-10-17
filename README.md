# News Topic Classifier

Routes a headline + short description to one of 9 sections, abstaining when unsure. Trained only on `news.csv` (from the News Category Dataset, Misra, CC BY 4.0; text © HuffPost). Pretrained encoder `all-MiniLM-L6-v2`, frozen. Written with Claude's assistance; I reviewed it and can explain every part.

## Data findings and decisions
- 25,000 rows, 9 valid labels. Imbalanced (LIFESTYLE 24.0%, EDUCATION 2.0%), so selection uses macro-F1 and `class_weight` was tuned on validation.
- 9.3% of rows have no description: kept; the API treats it as optional. 12 repeated texts: not de-duplicated.
- split_a is random. split_b is time-based (train 2012–16, test 2017–22) with a big class-mix shift (POLITICS 16.7% → 45.6%, LIFESTYLE 28.6% → 5.7%).
- Inputs are headline + description only. `date` is excluded: class mix varies strongly by year and split_b's test years never appear in train. `id`, `link`, `authors` are excluded too. Cleaning is whitespace-only.
- Validation = stratified 15% of each split's train rows; splits are never combined; tests check no test row reaches train/validation. Seed 42.

## Results (test, scored once; model locked on validation first)
| Model | split_a macro-F1 / acc | split_b macro-F1 / acc |
|---|---|---|
| TF-IDF + LR (baseline) | 0.685 / 0.755 | 0.560 / 0.711 |
| MiniLM embeddings + LR | **0.695 / 0.769** | **0.606 / 0.748** |

Per-class tables and confusion matrices: `results/`. Weakest classes: TECH & SCIENCE (F1 0.45 / 0.39) and BUSINESS & MONEY in split_b (0.38).
**Why the splits differ:** validation-to-test macro-F1 fell 0.005–0.012 on split_a but 0.088–0.110 on split_b. This fits the class-mix shift (classes that shrank, e.g. BUSINESS & MONEY, have precision far below recall); vocabulary or labelling drift is not ruled out. **I would trust split_b's ~0.61**, since real traffic arrives after the training period. Small test classes (62–80 rows in split_b) are noisy; no confidence intervals computed.

## Stronger approach: worth it?
Frozen encoder + LR gave +0.010 macro-F1 on split_a (within noise) and +0.046 on split_b (better on all 9 classes); no significance test run. Modestly worth it, mainly for the time-shifted case. Cost: needs torch and a ~90 MB model download.

## Errors
30 test errors sampled by confidence band (one annotator): 11 topic overlap, 7 WORLD/POLITICS boundary, 5 politics-heavy TV/celebrity items labelled ENTERTAINMENT (all split_b), 3 questionable labels, 2 too little text, 2 clear model misses. Counts describe the sample, not frequencies. Label ambiguity looks like the main ceiling; see `results/error_analysis_findings.md`.

## Serving, latency, cost
API serves the embedding model fitted on all split_a train rows (assumption; not compared with the split_b model). Mean 6.0 ms per request (p50 5.8, p95 8.3; 300 sequential requests, laptop CPU, in-process, no network) ≈ 6 s wall-clock per 1,000 predictions; $0 beyond local compute. Load/throughput not measured.

## Abstention
Threshold **0.50**, chosen on validation only: highest mean macro-F1 on answered cases among thresholds keeping coverage ≥ 80% on both splits (the 80% floor is my assumption; 70% would give 0.60, 90% would give 0.40). Validation: coverage 0.84, macro-F1 on answered 0.76. After-the-fact test check (threshold not changed): coverage 0.85 / 0.82, macro-F1 on answered 0.746 / 0.687 vs 0.695 / 0.606 with no abstention. Abstention catches only 36–38% of errors; confidence is a practical score, not a guarantee. The threshold came from models fitted on 85% of train; the served model uses 100%.

## One more week
Time-aware validation for split_b; bootstrap confidence intervals; per-class calibration; compare a larger encoder or fine-tuning; multi-annotator review of the confused label pairs; measure TF-IDF latency and load.

## Run it
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Linux CPU-only: first pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
bash scripts/reproduce.sh         # needs internet once for the encoder
uvicorn src.api:app --port 8000
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"headline": "Senate passes new budget bill", "short_description": "Lawmakers voted late on Thursday."}'
# -> {"category":"POLITICS","confidence":...,"abstain":false}; abstaining returns "category": null
```
Assumptions: `news.csv` is committed; trained models are rebuilt, not committed; `category` is null on abstain so no untrusted label is returned.