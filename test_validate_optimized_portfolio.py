import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from validate_optimized_portfolio import validate_portfolio


class OptimizedPortfolioValidationTests(unittest.TestCase):
    def _write_valid_fixture(self, directory, *, initial_applied="True"):
        directory = Path(directory)
        success = {
            "sample_id": 1,
            "dynamic_model_id": "group_weighted_multinomial_logistic_gam",
            "stock_count": 50,
            "wgan_initial_iteration_limit": 2000,
            "wgan_rolling_iteration_limit": 10,
            "forecast_observations_per_model": 2,
            "simulation_paths_per_date": 10000,
            "elapsed_seconds": 3600.0,
        }
        (directory / "SUCCESS.json").write_text(json.dumps(success), encoding="utf-8")

        losses = []
        for state in range(4):
            losses.append({
                "STATE": state,
                "UPDATE_TYPE": "initial",
                "UPDATE_APPLIED": initial_applied,
                "ITERATIONS_COMPLETED": 100,
                "ITERATION_LIMIT": 2000,
                "CRITIC_LOSS": 0.1,
                "GENERATOR_LOSS": 0.2,
                "EFFECTIVE_SAMPLE_SIZE": 100.0,
            })
        losses.append({
            "STATE": 0,
            "UPDATE_TYPE": "rolling",
            "UPDATE_APPLIED": "True",
            "ITERATIONS_COMPLETED": 2,
            "ITERATION_LIMIT": 10,
            "CRITIC_LOSS": 0.1,
            "GENERATOR_LOSS": 0.2,
            "EFFECTIVE_SAMPLE_SIZE": 80.0,
        })
        pd.DataFrame(losses).to_csv(directory / "wgan_loss_report.csv", index=False)

        dynamic = success["dynamic_model_id"]
        dates = ["2017-01-03", "2017-01-04"]
        forecasts = [
            {
                "DATE": date,
                "MODEL_ID": model,
                "REALIZED_RETURN": 0.001,
                "SIMULATION_MEAN": 0.0,
            }
            for date in dates
            for model in ["fixed_hmm_transition", dynamic, "normal"]
        ]
        pd.DataFrame(forecasts).to_csv(directory / "daily_risk_forecasts.csv", index=False)
        pd.DataFrame([{"SAMPLE_ID": 1, "SCORE": 0.1}]).to_csv(
            directory / "risk_scores.csv", index=False
        )

        state_counts = [
            {
                "DATE": date,
                "MODEL_ID": model,
                "STATE": state,
                "SIMULATED_COUNT": 2500,
            }
            for date in dates
            for model in ["fixed_hmm_transition", dynamic]
            for state in range(4)
        ]
        pd.DataFrame(state_counts).to_csv(
            directory / "simulated_state_counts.csv", index=False
        )

    def test_valid_production_fixture_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_valid_fixture(directory)
            report = validate_portfolio(directory, require_production_settings=True)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["forecast_dates"], 2)

    def test_false_string_is_not_treated_as_true(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_valid_fixture(directory, initial_applied="False")
            with self.assertRaisesRegex(ValueError, "initial regime WGAN"):
                validate_portfolio(directory, require_production_settings=True)


if __name__ == "__main__":
    unittest.main()
