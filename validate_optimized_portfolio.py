"""Validate completeness and numerical integrity of one optimized portfolio run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _strict_booleans(values, column_name):
    """Parse a CSV boolean column without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        examples = sorted(normalized[invalid].unique().tolist())[:3]
        raise ValueError(f"{column_name} contains invalid boolean values: {examples}")
    return normalized.eq("true")


def validate_portfolio(output_dir, *, require_production_settings=False):
    output_dir = Path(output_dir)
    required = {
        "success": output_dir / "SUCCESS.json",
        "losses": output_dir / "wgan_loss_report.csv",
        "forecasts": output_dir / "daily_risk_forecasts.csv",
        "scores": output_dir / "risk_scores.csv",
        "state_counts": output_dir / "simulated_state_counts.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required outputs: {missing}")

    with required["success"].open(encoding="utf-8") as handle:
        success = json.load(handle)
    losses = pd.read_csv(required["losses"])
    forecasts = pd.read_csv(required["forecasts"])
    scores = pd.read_csv(required["scores"])
    state_counts = pd.read_csv(required["state_counts"])

    required_loss_columns = {
        "STATE",
        "UPDATE_TYPE",
        "UPDATE_APPLIED",
        "ITERATIONS_COMPLETED",
        "ITERATION_LIMIT",
        "CRITIC_LOSS",
        "GENERATOR_LOSS",
        "EFFECTIVE_SAMPLE_SIZE",
    }
    required_forecast_columns = {"DATE", "MODEL_ID", "REALIZED_RETURN"}
    required_state_count_columns = {"DATE", "MODEL_ID", "STATE", "SIMULATED_COUNT"}
    for label, frame, columns in [
        ("loss report", losses, required_loss_columns),
        ("forecast report", forecasts, required_forecast_columns),
        ("state-count report", state_counts, required_state_count_columns),
    ]:
        missing_columns = sorted(columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{label} is missing columns: {missing_columns}")

    initial = losses[losses["UPDATE_TYPE"].eq("initial")]
    update_applied = _strict_booleans(losses["UPDATE_APPLIED"], "UPDATE_APPLIED")
    if len(initial) != 4 or not update_applied.loc[initial.index].all():
        raise ValueError("All four initial regime WGAN fits must be applied.")

    initial_numeric = initial[
        ["CRITIC_LOSS", "GENERATOR_LOSS", "EFFECTIVE_SAMPLE_SIZE"]
    ].to_numpy(dtype=float)
    if not np.isfinite(initial_numeric).all():
        raise ValueError("Initial WGAN fits contain non-finite diagnostics.")

    applied = losses[
        losses["UPDATE_TYPE"].eq("rolling")
        & update_applied
    ]
    applied_numeric = applied[
        ["CRITIC_LOSS", "GENERATOR_LOSS", "EFFECTIVE_SAMPLE_SIZE"]
    ].to_numpy(dtype=float)
    if not np.isfinite(applied_numeric).all():
        raise ValueError("Applied rolling updates contain non-finite diagnostics.")

    forecast_numeric = forecasts.select_dtypes(include="number").to_numpy(dtype=float)
    if not np.isfinite(forecast_numeric).all():
        raise ValueError("Daily risk forecasts contain non-finite values.")

    dynamic_model = str(success["dynamic_model_id"])
    expected_forecast_models = {"fixed_hmm_transition", dynamic_model, "normal"}
    actual_forecast_models = set(forecasts["MODEL_ID"].astype(str))
    if actual_forecast_models != expected_forecast_models:
        raise ValueError(
            "Forecast models differ from the required fixed/dynamic/normal set: "
            f"{sorted(actual_forecast_models)}"
        )
    forecast_date_count = int(forecasts["DATE"].nunique())
    expected_forecasts_per_model = int(success["forecast_observations_per_model"])
    forecast_counts = forecasts.groupby("MODEL_ID")["DATE"].nunique()
    if not forecast_counts.eq(expected_forecasts_per_model).all():
        raise ValueError(
            "Forecast date counts differ across models or from SUCCESS.json: "
            f"{forecast_counts.to_dict()}"
        )
    if forecasts.duplicated(["DATE", "MODEL_ID"]).any():
        raise ValueError("Daily risk forecasts contain duplicate model/date rows.")
    dates_by_model = {
        str(model_id): set(frame["DATE"].astype(str))
        for model_id, frame in forecasts.groupby("MODEL_ID")
    }
    reference_dates = next(iter(dates_by_model.values()))
    if any(dates != reference_dates for dates in dates_by_model.values()):
        raise ValueError("Forecast models do not cover exactly the same dates.")

    expected_paths = int(success["simulation_paths_per_date"])
    expected_simulation_models = {"fixed_hmm_transition", dynamic_model}
    actual_simulation_models = set(state_counts["MODEL_ID"].astype(str))
    if actual_simulation_models != expected_simulation_models:
        raise ValueError(
            "State-count models differ from the required fixed/dynamic set: "
            f"{sorted(actual_simulation_models)}"
        )
    if state_counts.duplicated(["DATE", "MODEL_ID", "STATE"]).any():
        raise ValueError("State-count output contains duplicate model/date/state rows.")
    state_rows_per_model_date = state_counts.groupby(["DATE", "MODEL_ID"])["STATE"].nunique()
    if not state_rows_per_model_date.eq(4).all():
        raise ValueError("Every simulated model/date must contain all four regimes.")
    simulated_totals = state_counts.groupby(
        ["DATE", "MODEL_ID"]
    )["SIMULATED_COUNT"].sum()
    if not simulated_totals.eq(expected_paths).all():
        raise ValueError("At least one model/date has an incorrect simulated path total.")
    simulation_dates = set(state_counts["DATE"].astype(str))
    if simulation_dates != reference_dates:
        raise ValueError("State-count output does not cover the complete forecast-date set.")

    iterations = applied["ITERATIONS_COMPLETED"].to_numpy(dtype=float)
    iteration_limit = int(success["wgan_rolling_iteration_limit"])
    if (iterations < 1).any() or (iterations > iteration_limit).any():
        raise ValueError("Applied rolling iteration counts are outside their valid range.")

    if require_production_settings:
        production_expectations = {
            "stock_count": (int(success["stock_count"]), 50),
            "simulation_paths_per_date": (expected_paths, 10_000),
            "wgan_initial_iteration_limit": (
                int(success["wgan_initial_iteration_limit"]),
                2_000,
            ),
            "wgan_rolling_iteration_limit": (iteration_limit, 10),
        }
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in production_expectations.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"Production settings do not match: {mismatches}")

    report = {
        "status": "valid",
        "sample_id": int(success["sample_id"]),
        "forecast_dates": forecast_date_count,
        "models": sorted(forecasts["MODEL_ID"].astype(str).unique().tolist()),
        "initial_iteration_limit": int(success["wgan_initial_iteration_limit"]),
        "rolling_iteration_limit": iteration_limit,
        "applied_rolling_updates": int(len(applied)),
        "skipped_rolling_updates": int(
            losses["UPDATE_TYPE"].eq("rolling").sum() - len(applied)
        ),
        "mean_rolling_iterations": float(iterations.mean()) if iterations.size else 0.0,
        "median_rolling_iterations": float(np.median(iterations)) if iterations.size else 0.0,
        "p95_rolling_iterations": float(np.quantile(iterations, 0.95)) if iterations.size else 0.0,
        "updates_reaching_iteration_limit_rate": float(
            np.mean(iterations >= iteration_limit)
        ) if iterations.size else 0.0,
        "simulation_paths_per_date": expected_paths,
        "score_rows": int(len(scores)),
        "elapsed_hours": float(success.get("elapsed_seconds", 0.0)) / 3600.0,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Portfolio output directory to validate.")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Require the calibrated 50-stock/10,000-path production settings.",
    )
    arguments = parser.parse_args()
    print(json.dumps(
        validate_portfolio(
            arguments.output_dir,
            require_production_settings=arguments.production,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()
