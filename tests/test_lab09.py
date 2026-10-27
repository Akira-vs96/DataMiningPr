"""Check Penguins data and the training boundaries of both ensemble methods."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate, train_test_split

from breast_cancer_labs.labs.lab09_ensemble_learning import (
    build_base_models,
    build_folds,
    build_stacking,
    fit_blending,
    ordered_probabilities,
)
from breast_cancer_labs.penguins_data import (
    DATA_MEMBER,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    SPECIES,
    load_penguins_dataset,
)


class PenguinsDatasetTests(unittest.TestCase):
    def test_supplied_archive_and_missing_values_are_preserved(self) -> None:
        data = load_penguins_dataset()
        self.assertEqual(data.shape, (1000, 7))
        self.assertEqual(data.species.value_counts().to_dict(), {"Adelie": 461, "Gentoo": 288, "Chinstrap": 251})
        self.assertEqual(int(data.isna().sum().sum()), 50)
        self.assertEqual(int(data.duplicated().sum()), 0)
        self.assertEqual(data.index.name, "source_row")
        self.assertNotIn("species", FEATURE_NAMES)

    def test_loader_rejects_invalid_target(self) -> None:
        data = load_penguins_dataset()
        for invalid in [None, "unknown"]:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                data.loc[0, "species"] = invalid
                path = Path(temporary) / "invalid.zip"
                with ZipFile(path, "w") as archive:
                    archive.writestr(DATA_MEMBER, data.to_csv(index=False))
                with self.assertRaises(ValueError):
                    load_penguins_dataset(path)


class EnsembleBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = load_penguins_dataset()
        cls.X = data[FEATURE_NAMES]
        cls.y = data.species

    def test_oof_imputation_is_fitted_only_on_fold_training_rows(self) -> None:
        # A different median in every fold makes full-train preprocessing detectable.
        X = self.X.iloc[:90].copy()
        y = self.y.iloc[:90]
        X.loc[:, "body_mass_g"] = np.arange(len(X), dtype=float) ** 2
        X.loc[X.index[::7], "body_mass_g"] = np.nan
        folds = list(build_folds().split(X, y))
        stack = build_stacking(build_base_models())
        result = cross_validate(stack.estimators[0][1], X, y, cv=folds, return_estimator=True)
        for estimator, (train_positions, valid_positions) in zip(result["estimator"], folds):
            imputer = estimator.named_steps["preprocessing"].named_transformers_["numeric"]
            expected = X.iloc[train_positions][NUMERIC_FEATURES].median().to_numpy()
            np.testing.assert_allclose(imputer.statistics_, expected)
            self.assertNotEqual(imputer.statistics_[-1], X.body_mass_g.median())
            self.assertFalse(set(train_positions) & set(valid_positions))

    def test_blending_keeps_base_training_statistics_and_class_order(self) -> None:
        X_base, X_valid, y_base, y_valid = train_test_split(
            self.X, self.y, test_size=0.30, random_state=42, stratify=self.y,
        )
        X_base, X_valid = X_base.copy(), X_valid.copy()
        X_base.loc[:, "body_mass_g"] = 100.0
        X_base.loc[X_base.index[0], "body_mass_g"] = np.nan
        X_valid.loc[:, "body_mass_g"] = 99999.0
        X_valid.loc[:, "island"] = "unseen_validation_island"
        templates = build_base_models()
        blend = fit_blending(templates, X_base, y_base, X_valid, y_valid)
        self.assertEqual(blend.validation_features.shape, (len(X_valid), 9))
        self.assertEqual(blend.validation_features.index.tolist(), X_valid.index.tolist())
        self.assertEqual(blend.meta_model.named_steps["standardscaler"].n_samples_seen_, len(X_valid))
        for (_, fitted), (_, template) in zip(blend.base_models, templates):
            preprocessing = fitted.named_steps["preprocessing"]
            self.assertEqual(preprocessing.named_transformers_["numeric"].statistics_[-1], 100.0)
            encoder = preprocessing.named_transformers_["categorical"].named_steps["onehotencoder"]
            self.assertNotIn("unseen_validation_island", encoder.categories_[0])
            self.assertFalse(hasattr(template.named_steps["preprocessing"], "transformers_"))
        probability = ordered_probabilities(blend, X_valid.iloc[:10])
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        predicted = np.asarray(SPECIES)[probability.argmax(axis=1)]
        np.testing.assert_array_equal(predicted, blend.predict(X_valid.iloc[:10]))

    def test_blending_rejects_shared_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_blending(build_base_models(), self.X.iloc[:50], self.y.iloc[:50], self.X.iloc[40:60], self.y.iloc[40:60])

    def test_stacking_leaves_test_only_categories_outside_preprocessing(self) -> None:
        X_train, X_test, y_train, _ = train_test_split(
            self.X.iloc[:150], self.y.iloc[:150], test_size=0.30, random_state=42, stratify=self.y.iloc[:150],
        )
        X_test = X_test.copy()
        X_test.loc[:, "island"] = "test_only_island"
        # One shallow tree keeps this boundary check small while exercising real OOF fitting.
        stack = build_stacking([("dt", clone(build_base_models()[0][1]))]).fit(X_train, y_train)
        encoder = stack.estimators_[0].named_steps["preprocessing"].named_transformers_["categorical"].named_steps["onehotencoder"]
        self.assertNotIn("test_only_island", encoder.categories_[0])
        probability = ordered_probabilities(stack, X_test)
        self.assertEqual(probability.shape, (len(X_test), 3))
        np.testing.assert_allclose(probability.sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
