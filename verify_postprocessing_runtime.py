"""Run the complete original-paper diagnostics on existing portfolio outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def verify_postprocessing(portfolio_dir, result_dir=None):
    portfolio_dir = Path(portfolio_dir).resolve()
    required = {
        "real_returns": portfolio_dir / "portfolio_real_returns.csv",
        "forecasts": portfolio_dir / "daily_risk_forecasts.csv",
        "state_counts": portfolio_dir / "simulated_state_counts.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Portfolio output is missing: {missing}")

    simulation_paths = sorted(
        portfolio_dir.glob("representative_simulated_returns_*.csv")
    )
    if len(simulation_paths) != 2:
        raise ValueError(
            "Expected exactly two representative fixed/dynamic simulation files, "
            f"found {len(simulation_paths)}."
        )

    real_returns = pd.read_csv(required["real_returns"], index_col=0, parse_dates=[0])
    if "STATE" not in real_returns:
        raise ValueError("portfolio_real_returns.csv does not contain STATE.")
    n_states = int(real_returns["STATE"].nunique())
    n_stocks = int(len(real_returns.columns) - 1)
    if n_states != 4 or n_stocks != 50:
        raise ValueError(
            f"Expected four regimes and 50 stocks, found {n_states} and {n_stocks}."
        )

    with Path("pipeline_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    forecast_start = str(config["data"]["forecast_start"])

    if result_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result_dir = (
            portfolio_dir.parent
            / "postprocessing_checks"
            / f"{portfolio_dir.name}_{timestamp}"
        )
    result_dir = Path(result_dir).resolve()
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Post-processing result directory is not empty: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Import only after the inexpensive input checks, so dependency failures are
    # reported clearly before the diagnostic functions begin their calculations.
    from p3_1_stat_prop import p3_1_stat_prop
    from p3_2_stat_prop_mv import p3_2_stat_prop_mv

    model_reports = []
    prefix = "representative_simulated_returns_"
    for simulation_path in simulation_paths:
        model_id = simulation_path.stem.removeprefix(prefix)
        arguments = {
            "n_states": n_states,
            "n_stocks": n_stocks,
            "stock_rets_path": required["real_returns"],
            "sim_rets_path": simulation_path,
            "output_dir": result_dir,
            "train_date": forecast_start,
            "output_suffix": model_id,
        }
        print(f"Running univariate diagnostics for {model_id}...", flush=True)
        p3_1_stat_prop(**arguments)
        print(f"Running multivariate diagnostics for {model_id}...", flush=True)
        p3_2_stat_prop_mv(**arguments)

        generated = sorted(result_dir.glob(f"*_{model_id}.*"))
        if len(generated) != 24:
            raise RuntimeError(
                f"Expected 24 diagnostic artifacts for {model_id}, found {len(generated)}."
            )
        empty = [path.name for path in generated if path.stat().st_size == 0]
        if empty:
            raise RuntimeError(f"Empty diagnostic artifacts for {model_id}: {empty}")
        model_reports.append({
            "model_id": model_id,
            "artifact_count": len(generated),
        })

    report = {
        "status": "valid",
        "portfolio_dir": str(portfolio_dir),
        "result_dir": str(result_dir),
        "n_states": n_states,
        "n_stocks": n_stocks,
        "models": model_reports,
        "total_artifacts": sum(item["artifact_count"] for item in model_reports),
    }
    with (result_dir / "POSTPROCESSING_CHECK.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio_dir", help="Existing incomplete portfolio directory.")
    parser.add_argument("--result-dir", default=None)
    arguments = parser.parse_args()
    verify_postprocessing(arguments.portfolio_dir, arguments.result_dir)


if __name__ == "__main__":
    main()
