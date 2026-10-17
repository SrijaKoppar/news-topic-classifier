# Error analysis: embedding model, test rows

Model: sentence embeddings + logistic regression (locked on validation).
Source: 30 test errors, 15 per split, 5 per confidence band (>=0.80, 0.50-0.79, <0.50),
drawn with seed 42. Groups were assigned by one annotator (author-drafted, then reviewed),
so they are judgements, not ground truth.

## Population facts (all test errors, not the sample)

| | split_a | split_b |
|---|---|---|
| Errors / test rows | 1226 / 5298 (23.1%) | 1261 / 5008 (25.2%) |
| Errors with confidence >= 0.80 | 252 (20.6%) | 198 (15.7%) |
| Errors caught by the 0.50 abstention threshold | 36.3% | 38.1% |

Largest confusion pairs (both directions summed, from the top-8 lists):
- split_a: HEALTH & WELLNESS <-> LIFESTYLE 192 (15.7% of errors); ENTERTAINMENT <-> LIFESTYLE 140 (11.4%);
  WORLD <-> POLITICS 142 (11.6%); ENTERTAINMENT <-> POLITICS 116 (9.5%)
- split_b: ENTERTAINMENT <-> POLITICS 273 (21.6%); WORLD <-> POLITICS 236 (18.7%)

## Cause groups in the sample (counts describe the sample only, not how common each cause is)

| Group | split_a | split_b | What it implies |
|---|---|---|---|
| topic_overlap | 6 | 5 | Taxonomy: several categories legitimately fit (health/lifestyle, sports/politics, business/world). A single-label scheme caps achievable accuracy. |
| world_politics_boundary | 5 | 2 | Taxonomy/labelling: unclear rule for foreign events involving politicians vs domestic politics with a foreign angle. |
| political_content_labelled_entertainment | 0 | 5 | Data: politics-heavy text about TV hosts, actors and pundits is labelled ENTERTAINMENT. The text itself is political. |
| label_questionable | 2 | 1 | Data: the given label looks odd on reading (e.g. a fashion piece labelled HEALTH). |
| insufficient_context | 1 | 1 | Data: empty or near-empty description; text alone cannot settle it. |
| model_miss | 1 | 1 | Model: the label looks right and the prediction looks wrong. Only 2 of 30. |

## What this suggests (hypotheses, not established causes)

1. In this sample, most errors have a defensible alternative label. Only 2 of 30 look like clear model
   mistakes under my reading. Because the sample is confidence-stratified and annotated by one person,
   this must not be quoted as an error-rate breakdown.
2. High-confidence errors are not rare: 20.6% (split_a) and 15.7% (split_b) of errors have
   confidence >= 0.80. Abstention at 0.50 catches only 36-38% of errors, so most errors are still
   answered. Confidence cannot fix label ambiguity, because the model is confidently picking one
   of two reasonable labels.
3. The ENTERTAINMENT -> POLITICS cluster in split_b (5 of 5 sampled examples in that group) may reflect
   a labelling convention (who is speaking vs what is discussed). I cannot confirm this from the data.
   Checking more examples of this confusion would test it.
4. HEALTH & WELLNESS <-> LIFESTYLE is the single largest confusion in split_a. The sampled examples
   suggest the boundary is blurry, so improving the model is unlikely to help much.
5. What would help, in order of plausibility: a clearer label definition for the confused pairs,
   multi-label or "either" handling for flagged pairs, and a human-review route for low-confidence cases.
   A larger encoder would not fix the label-ambiguity cases.

## Limitations
- One annotator; groups are my judgement on a sample of 30.
- The sample is not proportional to the error population, so group counts are not frequencies.
- Findings were not used to change the model, threshold or selection.