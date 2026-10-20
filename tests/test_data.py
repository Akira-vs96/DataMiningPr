"""Regression tests for the canonical WDBC data layer."""

from __future__ import annotations

import unittest

from breast_cancer_labs.data import (
    COLUMN_NAMES,
    FEATURE_NAMES,
    clean_dataset,
    load_raw_dataset,
    split_features_target,
)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = load_raw_dataset()
        cls.clean, cls.quality = clean_dataset(cls.raw)

    def test_expected_shape_and_schema(self) -> None:
        self.assertEqual(self.raw.shape, (569, 32))
        self.assertEqual(self.raw.columns.tolist(), COLUMN_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 30)

    def test_clean_data_is_complete(self) -> None:
        self.assertEqual(int(self.clean.isna().sum().sum()), 0)
        self.assertEqual(self.quality["duplicates_removed"], 0)
        self.assertEqual(set(self.clean["diagnosis"]), {"B", "M"})

    def test_target_encoding_and_class_counts(self) -> None:
        self.assertEqual(self.clean["diagnosis"].value_counts().to_dict(), {"B": 357, "M": 212})
        self.assertEqual(set(self.clean["diagnosis_binary"]), {0, 1})

    def test_model_matrix_excludes_id(self) -> None:
        X, y = split_features_target(self.clean)
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(y.shape, (569,))
        self.assertNotIn("id", X.columns)


if __name__ == "__main__":
    unittest.main()
