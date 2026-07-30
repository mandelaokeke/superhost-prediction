# Model card: Superhost readiness benchmark

## Summary

This project estimates whether a New York City listing row belongs to the current Superhost class. Histogram gradient boosting performed best among the models tested with host-disjoint data splits.

The score is best understood as **Superhost-class similarity in the February 2026 snapshot**, not the probability that a host will become a Superhost.

## Intended use

- Educational demonstration of an end-to-end tabular classification workflow.
- Exploratory ranking of non-Superhost rows for analyst review.
- Portfolio evidence of connecting ML outputs to a business workflow.
- Comparison of a historical neural network with simpler tabular baselines.

## Out-of-scope use

- Predicting future conversion without longitudinal validation.
- Fully automated outreach, eligibility, enforcement, or ranking decisions.
- Causal claims about which interventions will change host outcomes.
- Application to another geography or time period without revalidation.

## Data

- 36,445 raw New York City listing rows.
- 35,917 rows with a known target.
- 21,342 unique hosts.
- Scrape ID `20260213082241`.
- Last-scraped dates of February 13–14, 2026.

The exact download URL and license record were not preserved. The schema is consistent with an Inside Airbnb detailed-listings export, but that source remains unconfirmed.

## Evaluation design

The historical master's-project split was performed by listing. It contained:

- 1,211 hosts shared by train and validation;
- 1,175 hosts shared by train and test; and
- 546 hosts shared by validation and test.

The refreshed benchmark stratifies hosts by target and assigns each host—and all of that host's listings—to exactly one split:

| Split | Rows | Hosts | Positive prevalence |
|---|---:|---:|---:|
| Train | 25,287 | 14,939 | 19.2% |
| Validation | 5,742 | 3,201 | 19.8% |
| Test | 4,888 | 3,202 | 20.6% |

## Features and preprocessing

- 24 numeric and 5 categorical source features.
- Median imputation and standardization for numeric features.
- Most-frequent imputation and one-hot encoding for categorical features.
- 319 processed features.
- Five intended columns excluded because every value was missing.

The model does not use listing or host IDs as features.

## Model comparison

Metrics use the unseen-host test set and a validation-selected threshold. The selection rule maximizes recall subject to validation precision of at least 0.50.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500 | 0.206 | 0.206 | 1.000 | 0.341 | 0.500 |
| Logistic regression | 0.781 | 0.490 | 0.556 | 0.377 | 0.449 | 0.650 |
| Histogram gradient boosting | **0.907** | **0.684** | **0.625** | **0.735** | **0.676** | **0.811** |

The champion's Brier score is 0.098. Its confusion matrix at threshold 0.35 is:

| | Predicted non-Superhost | Predicted Superhost |
|---|---:|---:|
| Actual non-Superhost | 3,440 | 443 |
| Actual Superhost | 266 | 739 |

The historical ANN achieved 0.896 ROC-AUC on the overlapping listing-level split. It is retained as project history, not as the recommended model.

## Subgroup observations

For boroughs with at least 100 test rows and both classes, ROC-AUC ranged from 0.892 in Manhattan to 0.923 in Brooklyn. PR-AUC ranged from 0.593 to 0.813, reflecting both model behavior and different group prevalences. Staten Island did not meet the reporting threshold.

These diagnostics are not evidence of fairness. They do not cover intersectional groups, uncertainty intervals, temporal stability, or downstream decision effects.

## Limitations and risks

1. **Target mismatch:** the model observes current status rather than a future outcome.
2. **No temporal test:** host grouping prevents entity leakage but does not measure performance on a later snapshot.
3. **Uncalibrated probability:** Brier score is reported, but formal calibration has not been fitted or tested.
4. **Policy drift:** Superhost criteria and marketplace behavior may differ from the snapshot.
5. **Selection and geographic bias:** performance varies by borough, room type, property type, and host segment.
6. **Missing-feature behavior:** five intended inputs have no observed values.
7. **Recommendation validity:** action rules are illustrative and not experimentally validated.
8. **Incomplete provenance:** the exact source URL and license record still need recovery.

## Required validation before operational use

- Recover and verify the source and license.
- Obtain later snapshots and create an out-of-time conversion target.
- Calibrate scores and choose a threshold from explicit operating costs.
- Add uncertainty intervals and broader subgroup analysis.
- Use robust feature attribution and domain review.
- Establish human review, data retention, monitoring, and appeal processes.
