import unittest

import numpy as np
import pandas as pd

from src.train_baselines import (
    GROUP_COLUMN,
    TARGET_COLUMN,
    build_host_grouped_splits,
    convert_superhost_label,
    threshold_table,
)


class LabelCleaningTests(unittest.TestCase):
    def test_supported_labels(self) -> None:
        positives = ["t", "TRUE", "1", " yes "]
        negatives = ["f", "FALSE", "0", " no "]
        self.assertTrue(all(convert_superhost_label(value) == 1 for value in positives))
        self.assertTrue(all(convert_superhost_label(value) == 0 for value in negatives))
        self.assertTrue(np.isnan(convert_superhost_label(None)))


class GroupedSplitTests(unittest.TestCase):
    def test_hosts_never_cross_splits(self) -> None:
        rows = []
        for host_id in range(40):
            target = host_id % 2
            for listing_number in range((host_id % 3) + 1):
                rows.append(
                    {
                        GROUP_COLUMN: host_id,
                        TARGET_COLUMN: target,
                        "listing_number": listing_number,
                    }
                )
        frame = pd.DataFrame(rows)
        train, validation, test = build_host_grouped_splits(frame)

        train_hosts = set(train[GROUP_COLUMN])
        validation_hosts = set(validation[GROUP_COLUMN])
        test_hosts = set(test[GROUP_COLUMN])

        self.assertFalse(train_hosts & validation_hosts)
        self.assertFalse(train_hosts & test_hosts)
        self.assertFalse(validation_hosts & test_hosts)
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))


class ThresholdTests(unittest.TestCase):
    def test_threshold_respects_precision_floor(self) -> None:
        target = np.array([1, 1, 0, 0])
        probabilities = np.array([0.90, 0.55, 0.60, 0.10])
        rows, selected = threshold_table(target, probabilities)
        selected_row = next(row for row in rows if row["threshold"] == selected)
        self.assertGreaterEqual(selected_row["precision"], 0.50)


if __name__ == "__main__":
    unittest.main()
