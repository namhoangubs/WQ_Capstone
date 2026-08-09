"""Tests for causal Dynamic Scale filtering and fixed-baseline reuse."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_volatility import gjr_variance_recursion
from pipeline_wgan import _simulate_portfolio_returns
from run_pipeline import _existing_fixed_scores


class _GeneratedBatch:
    def __init__(self, values):
        self._values = values

    def numpy(self):
        return self._values


class _ConstantGenerator:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def __call__(self, latent, training=False):
        values = np.tile(self._values, (len(latent), 1))
        return _GeneratedBatch(values)


class DynamicScaleTests(unittest.TestCase):
    def test_conditional_variance_is_strictly_causal(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        changed = returns.copy()
        changed[2] = -0.30
        parameters = {
            "omega": 1e-6,
            "alpha": 0.08,
            "gamma": 0.10,
            "beta": 0.85,
            "variance_floor": 1e-10,
        }

        original_variance = gjr_variance_recursion(returns, **parameters)
        changed_variance = gjr_variance_recursion(changed, **parameters)

        np.testing.assert_allclose(original_variance[:3], changed_variance[:3])
        self.assertNotEqual(original_variance[3], changed_variance[3])

    def test_standardization_reconstructs_returns(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        variance = gjr_variance_recursion(
            returns,
            omega=1e-6,
            alpha=0.08,
            gamma=0.10,
            beta=0.85,
        )
        sigma = np.sqrt(variance)
        residuals = returns / sigma

        np.testing.assert_allclose(residuals * sigma, returns, rtol=1e-12, atol=1e-12)

    def test_simulations_are_rescaled_before_portfolio_aggregation(self):
        generators = {
            0: _ConstantGenerator([1.0, -2.0]),
            1: _ConstantGenerator([3.0, 4.0]),
        }
        sampled_states = np.array([0, 1, 0])
        latent = np.zeros((3, 2), dtype=float)
        weights = np.array([0.25, 0.75])
        sigma = np.array([0.02, 0.01])

        simulations, representative = _simulate_portfolio_returns(
            generators,
            sampled_states,
            latent,
            weights,
            conditional_sigma=sigma,
        )

        expected_state_0 = 0.25 * 0.02 + 0.75 * (-2.0 * 0.01)
        expected_state_1 = 0.25 * (3.0 * 0.02) + 0.75 * (4.0 * 0.01)
        np.testing.assert_allclose(
            simulations, [expected_state_0, expected_state_1, expected_state_0]
        )
        np.testing.assert_allclose(representative, [0.02, -0.02])

    def test_fixed_scores_are_imported_only_when_forecasts_are_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline" / "portfolio_01"
            output = root / "new" / "portfolio_01"
            baseline.mkdir(parents=True)
            output.mkdir(parents=True)
            dates = pd.to_datetime(["2017-01-03", "2017-01-04"])
            fixed_forecasts = pd.DataFrame({
                "DATE": dates,
                "MODEL_ID": ["fixed_hmm_transition"] * 2,
                "REALIZED_RETURN": [0.01, -0.02],
            })
            dynamic_forecasts = pd.DataFrame({
                "DATE": dates,
                "MODEL_ID": ["group_weighted_multinomial_logistic_gam"] * 2,
                "REALIZED_RETURN": [0.01, -0.02],
            })
            fixed_forecasts.to_csv(baseline / "daily_risk_forecasts.csv", index=False)
            dynamic_forecasts.to_csv(output / "daily_risk_forecasts.csv", index=False)
            pd.DataFrame({
                "MODEL_ID": ["fixed_hmm_transition", "normal"],
                "VALUE": [1.0, 2.0],
            }).to_csv(baseline / "risk_scores.csv", index=False)
            config = {
                "_project_root": root,
                "comparison": {
                    "import_existing_fixed_results": True,
                    "fixed_results_directory": "baseline",
                },
            }

            imported = _existing_fixed_scores(
                config, {"sample_id": 1, "output_dir": output}
            )

            self.assertEqual(imported["MODEL_ID"].tolist(), ["fixed_hmm_transition"])

    def test_fixed_import_rejects_different_realized_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline" / "portfolio_01"
            output = root / "new" / "portfolio_01"
            baseline.mkdir(parents=True)
            output.mkdir(parents=True)
            fixed = pd.DataFrame({
                "DATE": ["2017-01-03"],
                "MODEL_ID": ["fixed_hmm_transition"],
                "REALIZED_RETURN": [0.01],
            })
            changed = fixed.copy()
            changed["MODEL_ID"] = "group_weighted_multinomial_logistic_gam"
            changed["REALIZED_RETURN"] = 0.02
            fixed.to_csv(baseline / "daily_risk_forecasts.csv", index=False)
            changed.to_csv(output / "daily_risk_forecasts.csv", index=False)
            pd.DataFrame({
                "MODEL_ID": ["fixed_hmm_transition"], "VALUE": [1.0]
            }).to_csv(baseline / "risk_scores.csv", index=False)
            config = {
                "_project_root": root,
                "comparison": {
                    "import_existing_fixed_results": True,
                    "fixed_results_directory": "baseline",
                },
            }

            with self.assertRaisesRegex(ValueError, "realized returns differ"):
                _existing_fixed_scores(
                    config, {"sample_id": 1, "output_dir": output}
                )


if __name__ == "__main__":
    unittest.main()
