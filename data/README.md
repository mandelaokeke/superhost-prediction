# Data contract

The original dataset is intentionally not stored in this repository. To reproduce the notebook, provide these prepared files:

```text
data/Data/airbnb_train.csv
data/Data/airbnb_validation.csv
data/Data/airbnb_test.csv
```

The supplied data contains a New York City snapshot with scrape ID
`20260213082241` and last-scraped dates of February 13–14, 2026.

The historical prepared splits contain:

| Split | Rows | Columns | Superhost prevalence |
|---|---:|---:|---:|
| Train | 25,141 | 85 | 19.46% |
| Validation | 5,388 | 85 | 19.47% |
| Test | 5,388 | 85 | 19.45% |

## Required target

`host_is_superhost` must be present in every split. The notebook accepts the following encodings:

- positive: `t`, `true`, `1`, `yes`
- negative: `f`, `false`, `0`, `no`

Values are matched case-insensitively.

## Candidate features

The notebook uses the intersection of the following candidates and the columns available in the training file.

**Numeric**

- `host_response_rate`
- `host_acceptance_rate`
- `accommodates`
- `bathrooms`
- `bedrooms`
- `beds`
- `price`
- `minimum_nights`
- `maximum_nights`
- `availability_30`
- `availability_60`
- `availability_90`
- `availability_365`
- `number_of_reviews`
- `number_of_reviews_ltm`
- `reviews_per_month`
- `review_scores_rating`
- `review_scores_accuracy`
- `review_scores_cleanliness`
- `review_scores_checkin`
- `review_scores_communication`
- `review_scores_location`
- `review_scores_value`
- `calculated_host_listings_count`
- `calculated_host_listings_count_entire_homes`
- `calculated_host_listings_count_private_rooms`
- `calculated_host_listings_count_shared_rooms`

**Categorical**

- `host_response_time`
- `room_type`
- `property_type`
- `neighbourhood_cleansed`
- `host_identity_verified`
- `instant_bookable`
- `has_availability`

`id` and `host_id` are retained only for constructing the historical candidate queue; they are not model features.

The full `airbnb_main.csv` file contains 36,445 rows. Of these, 35,917 have a
known Superhost target and appear across the three prepared split files.

## Historical split warning

The prepared splits are listing-disjoint but not host-disjoint:

| Split pair | Shared hosts |
|---|---:|
| Train / validation | 1,211 |
| Train / test | 1,175 |
| Validation / test | 546 |

Use `src/train_baselines.py` for the refreshed host-grouped benchmark. The
historical files remain necessary for reproducing the original notebook output.

## Provenance status

The city and snapshot dates are recoverable from the supplied files. The exact
download URL, license record, and original preprocessing script were not
preserved. The column schema is consistent with an Inside Airbnb detailed
listings export, but that origin remains unconfirmed.

Do not substitute an arbitrary listing dataset and compare its results directly with the metrics in the README. Data distributions and Superhost program rules can change across locations and time.

## Privacy and version control

Raw data, exported predictions, and trained artifacts are ignored by Git. Do not commit host or listing identifiers in screenshots, CSV outputs, or notebook previews intended for a public portfolio.
