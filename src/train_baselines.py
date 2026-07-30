"""Train leakage-resistant baseline models for the Superhost project.

The historical CSV splits are preserved for audit purposes, but the benchmark in
this script rebuilds train/validation/test sets at the host level. This prevents
listings owned by the same host from appearing on both sides of an evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "host_is_superhost"
GROUP_COLUMN = "host_id"
RANDOM_STATE = 42

NUMERIC_CANDIDATES = [
    "host_response_rate",
    "host_acceptance_rate",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "price",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
]

CATEGORICAL_CANDIDATES = [
    "host_response_time",
    "room_type",
    "property_type",
    "neighbourhood_cleansed",
    "host_identity_verified",
    "instant_bookable",
    "has_availability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Data"),
        help="Directory containing airbnb_main.csv and the historical split files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline-results.json"),
        help="Destination for the reproducible benchmark results.",
    )
    return parser.parse_args()


def convert_superhost_label(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip().lower()
    if normalized in {"t", "true", "1", "yes"}:
        return 1
    if normalized in {"f", "false", "0", "no"}:
        return 0
    return np.nan


def clean_numeric_text(series: pd.Series, characters: str) -> pd.Series:
    cleaned = series.astype("string")
    for character in characters:
        cleaned = cleaned.str.replace(character, "", regex=False)
    return pd.to_numeric(cleaned.str.strip(), errors="coerce")


def clean_boolean(series: pd.Series) -> pd.Series:
    mapping = {
        "t": 1,
        "true": 1,
        "yes": 1,
        "1": 1,
        "f": 0,
        "false": 0,
        "no": 0,
        "0": 0,
    }
    return series.astype("string").str.strip().str.lower().map(mapping).astype("float64")


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].map(convert_superhost_label)
    frame = frame.dropna(subset=[TARGET_COLUMN, GROUP_COLUMN])
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")

    for column in ["host_response_rate", "host_acceptance_rate"]:
        if column in frame:
            frame[column] = clean_numeric_text(frame[column], "%,")
    if "price" in frame:
        frame["price"] = clean_numeric_text(frame["price"], "$,")
    for column in ["host_identity_verified", "instant_bookable", "has_availability"]:
        if column in frame:
            frame[column] = clean_boolean(frame[column])

    if "bathrooms" not in frame and "bathrooms_text" in frame:
        frame["bathrooms"] = pd.to_numeric(
            frame["bathrooms_text"].astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0],
            errors="coerce",
        )
    return frame


def build_host_grouped_splits(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    host_targets = frame[[GROUP_COLUMN, TARGET_COLUMN]].drop_duplicates()
    inconsistent = host_targets.groupby(GROUP_COLUMN)[TARGET_COLUMN].nunique()
    if (inconsistent > 1).any():
        raise ValueError("At least one host has inconsistent target labels.")

    host_targets = host_targets.drop_duplicates(subset=GROUP_COLUMN)
    train_hosts, holdout_hosts = train_test_split(
        host_targets,
        test_size=0.30,
        stratify=host_targets[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    validation_hosts, test_hosts = train_test_split(
        holdout_hosts,
        test_size=0.50,
        stratify=holdout_hosts[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    split_host_ids = {
        "train": set(train_hosts[GROUP_COLUMN]),
        "validation": set(validation_hosts[GROUP_COLUMN]),
        "test": set(test_hosts[GROUP_COLUMN]),
    }
    splits = {
        name: frame[frame[GROUP_COLUMN].isin(host_ids)].copy()
        for name, host_ids in split_host_ids.items()
    }
    assert not (split_host_ids["train"] & split_host_ids["validation"])
    assert not (split_host_ids["train"] & split_host_ids["test"])
    assert not (split_host_ids["validation"] & split_host_ids["test"])
    return splits["train"], splits["validation"], splits["test"]


def historical_overlap(data_dir: Path) -> dict[str, Any]:
    frames = {
        name: pd.read_csv(
            data_dir / f"airbnb_{name}.csv",
            usecols=["id", GROUP_COLUMN],
            low_memory=False,
        )
        for name in ["train", "validation", "test"]
    }
    host_sets = {
        name: set(frame[GROUP_COLUMN].dropna()) for name, frame in frames.items()
    }
    output: dict[str, Any] = {}
    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        shared = host_sets[left] & host_sets[right]
        output[f"{left}_{right}"] = {
            "shared_hosts": len(shared),
            f"{left}_affected_rows": int(frames[left][GROUP_COLUMN].isin(shared).sum()),
            f"{right}_affected_rows": int(frames[right][GROUP_COLUMN].isin(shared).sum()),
        }
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subgroup_diagnostics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    column: str,
    minimum_rows: int = 100,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    values = frame[column].fillna("<missing>").astype(str).to_numpy()
    for value in sorted(set(values)):
        positions = np.flatnonzero(values == value)
        subgroup_target = y_true[positions]
        if len(positions) < minimum_rows or len(np.unique(subgroup_target)) < 2:
            continue
        subgroup_metrics = metrics(
            subgroup_target,
            probabilities[positions],
            threshold,
        )
        diagnostics[value] = {
            "rows": len(positions),
            "positive_prevalence": float(subgroup_target.mean()),
            **subgroup_metrics,
        }
    return diagnostics


def threshold_table(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[list[dict[str, float]], float]:
    rows: list[dict[str, float]] = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
            }
        )
    eligible = [row for row in rows if row["precision"] >= 0.50]
    selected = max(eligible or rows, key=lambda row: row["recall"] if eligible else row["f1"])
    return rows, float(selected["threshold"])


def metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    print("Loading and cleaning the main dataset...", flush=True)
    raw = pd.read_csv(args.data_dir / "airbnb_main.csv", low_memory=False)
    frame = clean_frame(raw)
    train, validation, test = build_host_grouped_splits(frame)

    present_numeric = [column for column in NUMERIC_CANDIDATES if column in train]
    present_categorical = [
        column for column in CATEGORICAL_CANDIDATES if column in train
    ]
    all_missing = [
        column
        for column in present_numeric + present_categorical
        if train[column].notna().sum() == 0
    ]
    numeric = [column for column in present_numeric if column not in all_missing]
    categorical = [
        column for column in present_categorical if column not in all_missing
    ]
    features = numeric + categorical

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )

    print(
        f"Fitting preprocessing on {len(train):,} rows and {len(features)} source features...",
        flush=True,
    )
    x_train = preprocessor.fit_transform(train[features])
    x_validation = preprocessor.transform(validation[features])
    x_test = preprocessor.transform(test[features])
    y_train = train[TARGET_COLUMN].to_numpy()
    y_validation = validation[TARGET_COLUMN].to_numpy()
    y_test = test[TARGET_COLUMN].to_numpy()

    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1_000,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),
    }

    results: dict[str, Any] = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        test_probabilities = model.predict_proba(x_test)[:, 1]
        table, selected_threshold = threshold_table(
            y_validation, validation_probabilities
        )
        results[name] = {
            "selected_threshold": selected_threshold,
            "validation_threshold_table": table,
            "test_at_selected_threshold": metrics(
                y_test, test_probabilities, selected_threshold
            ),
            "test_at_default_threshold": metrics(y_test, test_probabilities, 0.50),
        }
        if name == "hist_gradient_boosting":
            results[name]["subgroup_diagnostics"] = {
                column: subgroup_diagnostics(
                    test.reset_index(drop=True),
                    y_test,
                    test_probabilities,
                    selected_threshold,
                    column,
                )
                for column in ["neighbourhood_group_cleansed", "room_type"]
            }

    feature_names = preprocessor.get_feature_names_out()
    coefficients = models["logistic_regression"].coef_[0]
    order = np.argsort(coefficients)
    logistic_coefficients = {
        "most_negative": [
            {
                "feature": str(feature_names[index]),
                "coefficient": float(coefficients[index]),
            }
            for index in order[:15]
        ],
        "most_positive": [
            {
                "feature": str(feature_names[index]),
                "coefficient": float(coefficients[index]),
            }
            for index in order[-15:][::-1]
        ],
    }

    split_summary = {
        name: {
            "rows": len(split),
            "hosts": int(split[GROUP_COLUMN].nunique()),
            "positive_prevalence": float(split[TARGET_COLUMN].mean()),
        }
        for name, split in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]
    }
    payload = {
        "benchmark_version": 1,
        "random_state": RANDOM_STATE,
        "environment": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "rows_before_target_filter": len(raw),
            "rows_after_target_filter": len(frame),
            "unique_hosts": int(frame[GROUP_COLUMN].nunique()),
            "scrape_id": [int(value) for value in raw["scrape_id"].dropna().unique()],
            "last_scraped": sorted(
                raw["last_scraped"].dropna().astype(str).unique().tolist()
            ),
            "neighbourhood_groups": {
                str(key): int(value)
                for key, value in raw["neighbourhood_group_cleansed"]
                .value_counts()
                .items()
            },
            "file_sha256": {
                path.name: sha256(path)
                for path in sorted(args.data_dir.glob("airbnb_*.csv"))
            },
        },
        "historical_split_overlap": historical_overlap(args.data_dir),
        "grouped_split": split_summary,
        "feature_summary": {
            "numeric": numeric,
            "categorical": categorical,
            "all_missing_excluded": all_missing,
            "source_feature_count": len(features),
            "processed_feature_count": int(x_train.shape[1]),
        },
        "models": results,
        "logistic_coefficients": logistic_coefficients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved benchmark results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
