"""Tests for causal adaptive regime memory and ESS gating inputs."""

import unittest

import numpy as np
import pandas as pd

from pipeline_wgan import (
    _adaptive_regime_memory,
    _effective_sample_size,
    _fit_wgan_window,
)


def _settings():
    return {
        "rolling_window_size": 256,
        "latent_dimension": 4,
        "training_iterations": 2,
        "batch_size": 2,
        "critic_steps": 1,
        "gradient_penalty": 10.0,
        "regime_memory": {
            "weighting": "exponential",
            "recency_half_life_trading_days": 256,
            "minimum_effective_sample_size": 64,
        },
    }


class AdaptiveRegimeMemoryTests(unittest.TestCase):
    def test_effective_sample_size_matches_uniform_sample_count(self):
        self.assertAlmostEqual(_effective_sample_size(np.ones(64)), 64.0)
        self.assertEqual(_effective_sample_size([]), 0.0)

    def test_abundant_regime_uses_only_base_window(self):
        index = pd.bdate_range("2000-01-03", periods=400)
        stock = pd.DataFrame(
            {"AAA": np.arange(400, dtype=float), "STATE": np.zeros(400, dtype=int)},
            index=index,
        )

        memory = _adaptive_regime_memory(
            stock, 400, 0, ["AAA"], _settings()
        )

        self.assertEqual(memory["metadata"]["BASE_WINDOW_OBSERVATIONS"], 256)
        self.assertEqual(memory["metadata"]["MEMORY_LOOKBACK_ROWS"], 256)
        self.assertGreaterEqual(memory["metadata"]["EFFECTIVE_SAMPLE_SIZE"], 64)
        self.assertEqual(memory["data"][0, 0], 144.0)

    def test_sparse_regime_expands_beyond_base_window_until_ess_threshold(self):
        index = pd.bdate_range("2000-01-03", periods=800)
        states = np.ones(800, dtype=int)
        states[::8] = 0
        stock = pd.DataFrame(
            {"AAA": np.arange(800, dtype=float), "STATE": states}, index=index
        )

        memory = _adaptive_regime_memory(
            stock, 800, 0, ["AAA"], _settings()
        )

        metadata = memory["metadata"]
        self.assertEqual(metadata["BASE_WINDOW_OBSERVATIONS"], 32)
        self.assertGreater(metadata["MEMORY_LOOKBACK_ROWS"], 256)
        self.assertGreaterEqual(metadata["EFFECTIVE_SAMPLE_SIZE"], 64)
        self.assertGreaterEqual(len(memory["data"]), 64)

    def test_insufficient_history_remains_below_gate(self):
        index = pd.bdate_range("2000-01-03", periods=400)
        states = np.ones(400, dtype=int)
        states[:50] = 0
        stock = pd.DataFrame(
            {"AAA": np.arange(400, dtype=float), "STATE": states}, index=index
        )

        memory = _adaptive_regime_memory(
            stock, 400, 0, ["AAA"], _settings()
        )

        self.assertEqual(len(memory["data"]), 50)
        self.assertLess(memory["metadata"]["EFFECTIVE_SAMPLE_SIZE"], 64)

    def test_fit_skips_update_below_ess_gate(self):
        result = _fit_wgan_window(
            state=0,
            state_data=np.zeros((10, 2), dtype=np.float32),
            sample_weights=np.ones(10),
            networks={0: (object(), object())},
            optimizers={0: (object(), object())},
            settings=_settings(),
            window_number=1,
            rng=np.random.default_rng(42),
            components=(None, None, None, None, None, None, None),
        )

        self.assertFalse(result["UPDATE_APPLIED"])
        self.assertEqual(result["ITERATIONS_COMPLETED"], 0)
        self.assertEqual(result["UPDATE_REASON"], "ess_below_threshold")

    def test_memory_is_strictly_pre_forecast(self):
        index = pd.bdate_range("2000-01-03", periods=500)
        stock = pd.DataFrame(
            {"AAA": np.arange(500, dtype=float), "STATE": np.zeros(500, dtype=int)},
            index=index,
        )
        forecast_position = 420

        memory = _adaptive_regime_memory(
            stock, forecast_position, 0, ["AAA"], _settings()
        )

        self.assertLess(memory["data"][:, 0].max(), forecast_position)
        self.assertEqual(
            pd.Timestamp(memory["metadata"]["MEMORY_END_DATE"]),
            index[forecast_position - 1],
        )


if __name__ == "__main__":
    unittest.main()
