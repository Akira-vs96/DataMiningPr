"""Boundary checks for forecasting and finite-sample calibration, variant 1."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from breast_cancer_labs.labs.lab10_time_series import seasonal_naive
from breast_cancer_labs.labs.lab12_concept_drift import alarm_summary
from breast_cancer_labs.labs.lab14_topic_modeling import restore_abstract
from breast_cancer_labs.labs.lab15_prediction_intervals import conformal_quantile
from breast_cancer_labs.practical_common import electricity_features, load_daily_electricity


class PracticalBoundaryTests(unittest.TestCase):
    def test_fixed_forecast_repeats_only_last_training_season(self):
        train = pd.Series(np.arange(36))
        result = seasonal_naive(train, 25)
        np.testing.assert_array_equal(result, np.r_[np.arange(24, 36), np.arange(24, 36), 24])
        self.assertEqual(train.iloc[-1], 35)
        with self.assertRaises(ValueError):
            seasonal_naive(train.iloc[:11], 12)

    def test_conformal_uses_corrected_order_statistic(self):
        # The finite-sample correction differs from the naive 80th percentile.
        self.assertEqual(conformal_quantile(np.arange(1., 11.), .2), 9.)
        self.assertEqual(conformal_quantile(np.arange(10., 0., -1.), .1), 10.)

    def test_small_calibration_can_require_infinite_interval(self):
        self.assertTrue(np.isinf(conformal_quantile(np.array([1., 2., 3.]), .05)))
        for invalid in [np.array([]), np.array([np.nan]), np.array([-1.])]:
            with self.assertRaises(ValueError):
                conformal_quantile(invalid, .1)
        for alpha in [0., 1.]:
            with self.assertRaises(ValueError):
                conformal_quantile(np.array([1.]), alpha)

    def test_alarm_boundaries_include_change_and_horizon(self):
        summary = alarm_summary([199, 200, 300, 301], change=200, horizon=100)
        self.assertEqual(summary["false_before_change"], 1)
        self.assertEqual(summary["delay"], 0)
        self.assertEqual(summary["late_alarms"], 1)
        self.assertIsNone(alarm_summary([199, 301], change=200, horizon=100)["delay"])

    def test_abstract_restoration_preserves_word_positions_and_repetitions(self):
        self.assertEqual(restore_abstract({"energy": [0, 3], "and": [1], "climate": [2]}), "energy and climate energy")

    def test_weather_calendar_features_do_not_use_target_or_future_rows(self):
        data = load_daily_electricity()
        self.assertEqual(len(data), 365)
        self.assertTrue(data.index.is_monotonic_increasing)
        features = electricity_features(data)
        changed = data.copy()
        changed["Demand"] = -999
        pd.testing.assert_frame_equal(features, electricity_features(changed))
        pd.testing.assert_frame_equal(features.iloc[:60], electricity_features(data.iloc[:60]))


if __name__ == "__main__":
    unittest.main()
