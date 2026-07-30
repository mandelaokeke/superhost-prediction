# Leakage-resistant benchmark

## Executive summary

The original notebook used disjoint listing rows, but not disjoint hosts. In the historical split, 1,175 hosts appeared in both training and test data, affecting 7,955 training rows and 2,535 test rows.

The refreshed benchmark assigns every listing from a host to exactly one split. On the resulting unseen-host test set, histogram gradient boosting achieved:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.907 |
| PR-AUC | 0.684 |
| Accuracy | 0.855 |
| Precision | 0.625 |
| Recall | 0.735 |
| F1 | 0.676 |
| Balanced accuracy | 0.811 |
| Brier score | 0.098 |
| Validation-selected threshold | 0.35 |

The confusion matrix at the selected threshold was:

| | Predicted non-Superhost | Predicted Superhost |
|---|---:|---:|
| Actual non-Superhost | 3,440 | 443 |
| Actual Superhost | 266 | 739 |

## Dataset

- New York City listing snapshot
- Scrape ID `20260213082241`
- Last-scraped dates: February 13–14, 2026
- 36,445 raw listing rows
- 35,917 rows with a known target
- 21,342 unique hosts
- Five borough groups represented

The exact original download URL and license record were not preserved. The schema is consistent with an Inside Airbnb detailed-listings export, but that origin should remain marked as unconfirmed until the source record is recovered.

## Split design

The target is constant within each host in this snapshot. Hosts were stratified by target and divided into:

| Split | Rows | Hosts | Positive prevalence |
|---|---:|---:|---:|
| Train | 25,287 | 14,939 | 19.2% |
| Validation | 5,742 | 3,201 | 19.8% |
| Test | 4,888 | 3,202 | 20.6% |

There is zero host overlap across the refreshed splits. Validation and test row counts vary because hosts own different numbers of listings.

## Model comparison

Metrics below use the test set and each model's validation-selected threshold. The selection rule maximizes recall while requiring at least 0.50 validation precision.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Dummy prior | 0.500 | 0.206 | 0.206 | 1.000 | 0.341 | 0.500 |
| Logistic regression | 0.781 | 0.490 | 0.556 | 0.377 | 0.449 | 0.650 |
| Histogram gradient boosting | **0.907** | **0.684** | **0.625** | **0.735** | **0.676** | **0.811** |

The boosted-tree model is the strongest benchmark. It also surpasses the historical ANN's 0.896 ROC-AUC while being evaluated under the stricter unseen-host design. PR-AUC is not directly comparable because the refreshed test prevalence and membership differ.

## Input-quality findings

Five intended fields contain no observed values and are excluded before preprocessing:

- `host_response_rate`
- `host_acceptance_rate`
- `price`
- `host_response_time`
- `instant_bookable`

The final benchmark uses 29 source features expanded to 319 processed features.

## Subgroup diagnostics

The champion model was evaluated by borough and room type for groups with at least 100 test rows and both target classes.

| Borough | Rows | ROC-AUC | PR-AUC | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Bronx | 171 | 0.920 | 0.813 | 0.618 | 0.829 |
| Brooklyn | 1,904 | 0.923 | 0.745 | 0.671 | 0.766 |
| Manhattan | 2,057 | 0.892 | 0.593 | 0.561 | 0.668 |
| Queens | 698 | 0.898 | 0.709 | 0.653 | 0.778 |

Staten Island was excluded from the table because it had fewer than 100 test rows. These are diagnostics, not fairness guarantees: group prevalence and sample size differ, and neighborhood is itself included as a model feature.

## Reproduce

```bash
python src/train_baselines.py
```

The machine-readable results, threshold tables, data hashes, package versions, subgroup metrics, and logistic-regression coefficients are stored in [`baseline-results.json`](baseline-results.json).
