# DSA 2050A — Week 9: Model selection and defensible evaluation

**Group:** Group 7 
**Submission:** One shared GitHub repository link per group  
**Course:** DSA 2050A Data Science Methodology  
**Week:** 9

## Group responsibilities
Complete the lab **in your existing groups**. Work together on one notebook and one repository. Record each member's concrete contribution in the table below. Each member should contribute visibly through Git commits or documented co-authored work.

| Group member | Student ID | Work completed | GitHub username or commit reference |
|---|---|---|---|
|  Moses Ogenrwot | 673380 | [Example: evaluation pipeline] | [Username] |
|  Susan Wnajiru Nduta | 673011 | [Example: error analysis and README] | [Username] |
| Rachel Momita |    |             |                              |         |
| Sylvia Kagwwka |     |       |         |

## Study objective
A fictional Kenyan internet provider wants to identify customers at risk of churn so it can offer retention deals before they leave. This lab compares two candidate models fairly, investigates where they fail, and records the model choice **before** the final test is read. Model selection must use development data only because the test set is our one unbiased estimate of performance on unseen customers. If test results influence any decision, the reported score becomes optimistic and no longer reflects real-world behaviour.

## Data statement
Synthetic fictional customer data created for teaching. No real customer records.

## Setup and reproduction
1. Open this folder in VS Code.
2. Create/select a Python environment (3.10–3.13 recommended).
3. In **Terminal → New Terminal**, run `python -m pip install -r requirements.txt`.
4. Open `notebooks/Week9_Model_Selection_Lab.ipynb` and **Select Kernel**.
5. Run each cell in order. Complete the rationale before the final-test cell.

## Prediction-time features and excluded fields
- Target: `churn` (1 = left; 0 = stayed).
- Excluded: `customer_id` (identifier), `cancellation_confirmed` (post-outcome leakage).
- Allowed features (all known before a customer cancels): `tenure_months`, `monthly_charges_kes`, `support_tickets_90d`, `service_calls_90d`, `plan_type`, `region`, `autopay`, `customer_segment`. Numeric features are median-imputed and scaled; categorical features are mode-imputed and one-hot encoded, all inside a `Pipeline` so preprocessing is re-fit within each CV fold (no leakage).
- Why exclusions: `cancellation_confirmed` is only known after the outcome (leakage); `customer_id` is an arbitrary identifier with no predictive meaning.

## Experiments and model selection
- Primary metric chosen before comparison: **F1**.
- CV: **5-fold stratified, shuffle=True, seed=42**.
- Naive baseline (majority class): mean F1 = 0.000 (it never predicts churn; accuracy would be about 66.5%).

Candidate comparison (from `reports/experiment_log.csv`, development data only):

| Model | Setting | Mean F1 | SD F1 | Mean precision | Mean recall |
|---|---|---|---|---|---|
| Logistic regression | C = 1 | 0.470 | 0.053 | 0.624 | 0.378 |
| Logistic regression | C = 10 | 0.470 | 0.053 | 0.624 | 0.378 |
| Logistic regression | C = 0.1 | 0.465 | 0.054 | 0.625 | 0.371 |
| Decision tree | max_depth = 3 | 0.398 | 0.049 | 0.566 | 0.318 |
| Decision tree | max_depth = 5 | 0.382 | 0.023 | 0.583 | 0.287 |
| Decision tree | max_depth = 8 | 0.372 | 0.051 | 0.495 | 0.300 |

- **Chosen model:** Logistic regression
- **Tuned hyperparameters:** `C = 1` (C = 10 tied; the more regularised value was kept)
- **Mean/SD CV F1:** 0.470 / 0.053
- **Why this choice:** Logistic regression had the highest mean F1, our primary metric, and beat the best decision tree (0.398, SD 0.049) by about 0.07, which is larger than the fold-to-fold noise of about 0.05. It also had higher precision and recall than the tree. Both models had similar stability, so there was no stability argument for overriding the highest-F1 default. Logistic regression is simpler and easier to explain to a business audience, and its regularisation limits overfitting, whereas the tree's score fell as depth increased (0.398, 0.382, 0.372), a sign of overfitting. The choice and written rationale were saved to `reports/frozen_selection.json` before the test file was read.

## Error and subgroup analysis
- Out-of-fold confusion matrix on the development data: 1,810 actual stays / 910 actual churners. The model produced **566 false negatives** (missed churners) and **205 false positives** (loyal customers flagged). Overall churn recall is 0.38 and precision is 0.63.
- Subgroup recall (from `reports/subgroup_metrics.csv`):

| Segment | Customers | Actual churners | Recall | Precision | False negatives |
|---|---|---|---|---|---|
| Established | 2,268 | 649 | 0.216 | 0.593 | 509 |
| New | 452 | 261 | 0.782 | 0.652 | 57 |

- Contrary to the lab's hypothesis, recall is **worse for established customers (0.22) than for new customers (0.78)**. Both groups have a reasonable number of actual churners, so this is unlikely to be a small-sample artefact.
- Business implication: a retention team using this model would miss about 78% of churning established customers, who make up most of the customer base and account for 509 of the 566 missed churners. Missed churners cost revenue without any chance to intervene; false positives only cost the price of an unneeded offer. The model should not be relied on for established customers without improvement (for example a lower decision threshold or extra features).

## Permutation importance
- Computed on a 25% validation slice of the development data (5 repeats, scoring F1). Top two features:
  1. `tenure_months`: mean F1 drop of about 0.079
  2. `support_tickets_90d`: mean F1 drop of about 0.035
- Permutation importance shows only what the fitted model relies on for prediction. It does not show that changing tenure or tickets would change churn: correlated features (e.g. tenure and customer segment) share credit, the estimates are noisy on a small validation slice, and unobserved factors may drive both the feature and the outcome.

## Final test (run once only)
- Evaluated once on 680 customers (`reports/final_metrics.json`), after refitting the frozen model on all development data.
- F1: **0.449** ; precision: **0.585** ; recall: **0.364** ; accuracy: 0.700
- Confusion matrix (rows = actual, cols = predicted; 0 = stay, 1 = churn): `[[393, 59], [145, 83]]`
- CV vs test: mean CV F1 was 0.470 and test F1 was 0.449, a drop of about 0.02. This is well within one fold standard deviation (0.053), so it is not concerning and is consistent with the small optimism expected from selecting on development data. The model generalised reasonably, but it still misses about 64% of churners.

## Limitations and ethics
- **Synthetic data:** relationships (e.g. the strong tenure effect) may be simpler than in a real customer base, so these numbers cannot be assumed to transfer to a real provider.
- **Hidden customer dynamics and no time dimension:** the data is a single snapshot. Drivers such as competitor offers, network quality, billing disputes and seasonality are not recorded, and we cannot test temporal drift.
- **Uneven performance:** recall is much lower for established customers, so the model is not equally useful across groups.
- **Before real use:** run an out-of-time validation, pilot with human review of flagged customers (retention offers should not be applied automatically), monitor recall by subgroup and overall drift after launch, and review the decision threshold with the business based on the relative cost of missed churners versus unnecessary offers.

## Evidence checklist
- [x] Executed notebook with explanations
- [x] `reports/experiment_log.csv`
- [x] `reports/subgroup_metrics.csv`
- [x] `reports/permutation_importance.csv`
- [x] `reports/frozen_selection.json` created before test
- [x] `reports/final_metrics.json` generated once
- [x] Two generated PNG charts
- [ ] Meaningful commits or co-authorship evidence for all group members
- [ ] Completed group contribution table with names and student IDs

## Final group review
- [x] One team member verified the notebook runs from start to finish
- [ ] Group discussed and approved the written model-selection rationale
- [x] No one used final-test scores to revise the model
- [ ] One representative will submit the shared URL to the course LMS

## GitHub repository URL
[Paste URL here]
