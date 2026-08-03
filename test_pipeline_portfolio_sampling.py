"""Tests for the balanced low-overlap portfolio sampling design."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline_runtime import (
    _balanced_low_overlap_assignment,
    _pairwise_overlap_matrix,
    create_or_load_portfolio_manifests,
)


class BalancedPortfolioSamplingTests(unittest.TestCase):
    def setUp(self):
        self.universe = [f"STOCK_{number:03d}" for number in range(183)]

    def _assignment(self, seed=42, attempts=30, swaps=5000):
        return _balanced_low_overlap_assignment(
            self.universe,
            sample_count=10,
            stocks_per_sample=50,
            rng=np.random.default_rng(seed),
            construction_attempts=attempts,
            pairwise_swap_iterations=swaps,
        )[0]

    def test_balanced_assignment_has_exact_sizes_and_inclusion_counts(self):
        assignment = self._assignment()
        frequencies = assignment.sum(axis=1)
        overlaps = _pairwise_overlap_matrix(assignment)
        pairwise = overlaps[np.triu_indices(10, k=1)]

        self.assertTrue(np.all(assignment.sum(axis=0) == 50))
        self.assertEqual(int((frequencies == 2).sum()), 49)
        self.assertEqual(int((frequencies == 3).sum()), 134)
        self.assertEqual(int(pairwise.min()), 10)
        self.assertEqual(int(pairwise.max()), 11)
        self.assertAlmostEqual(float(pairwise.mean()), 451 / 45)

    def test_balanced_assignment_is_reproducible_for_a_fixed_seed(self):
        first = self._assignment(42, attempts=1, swaps=0)
        second = self._assignment(42, attempts=1, swaps=0)
        different_seed = self._assignment(43, attempts=1, swaps=0)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, different_seed))

    def test_manifest_writer_persists_overlap_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "_project_root": directory,
                "portfolio_sampling": {
                    "number_of_samples": 10,
                    "stocks_per_sample": 50,
                    "sampling_design": "balanced_low_overlap",
                    "balanced_construction_attempts": 1,
                    "pairwise_swap_iterations": 0,
                    "random_seed": 42,
                    "weighting": "equal",
                    "manifest_directory": "manifests",
                    "reuse_existing_manifests": False,
                },
            }
            with patch("pipeline_runtime.eligible_stock_universe", return_value=self.universe):
                manifests = create_or_load_portfolio_manifests(config, write=True)

            manifest_directory = Path(directory) / "manifests"
            self.assertEqual(len(manifests), 10)
            self.assertTrue((manifest_directory / "portfolio_overlap_matrix.csv").exists())
            self.assertTrue(
                (manifest_directory / "portfolio_stock_inclusion_frequency.csv").exists()
            )
            with (manifest_directory / "portfolio_sampling_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            self.assertEqual(summary["sampling_design"], "balanced_low_overlap")
            self.assertEqual(summary["minimum_stock_inclusion_count"], 2)
            self.assertEqual(summary["maximum_stock_inclusion_count"], 3)
            self.assertLessEqual(
                summary["pairwise_overlap_maximum"]
                - summary["pairwise_overlap_minimum"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
