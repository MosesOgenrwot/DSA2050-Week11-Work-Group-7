# Model-selection rationale

**Selected:** Logistic regression, C = 1 (primary metric F1, 5-fold stratified CV, shuffle, seed 42).

- Mean CV F1 0.470 (SD 0.053) vs best decision tree 0.398 (SD 0.049); better precision (0.624 vs 0.566) and recall (0.378 vs 0.318).
- Similar fold variability, so the ~0.07 gap is not explained by noise.
- Simpler and easier to explain; the tree's score fell as depth grew (0.398, 0.382, 0.372).
- Out-of-fold errors: 566 false negatives, 205 false positives. Recall is 0.22 for Established (649 churners) and 0.78 for New customers (261 churners).
- Selection frozen in `frozen_selection.json` before the single final test (F1 0.449, precision 0.585, recall 0.364).
