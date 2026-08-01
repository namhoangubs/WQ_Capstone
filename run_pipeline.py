"""Config-driven q4 HMM-WGAN experiment runner."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline_phase1 import run_phase_1_production
from pipeline_risk import (
    aggregate_paired_scores,
    save_p_value_figures,
    save_scorecard_figures,
    summarize_backtest_p_values,
)
from pipeline_runtime import (
    build_execution_plan,
    create_or_load_portfolio_manifests,
    load_pipeline_config,
    resolve_project_path,
    validate_selected_registry,
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def _aggregate_completed_outputs(config, results):
    score_frames = []
    for result in results:
        path = Path(result["scores_path"])
        if path.exists():
            score_frames.append(pd.read_csv(path))
    if not score_frames:
        return None
    output_root = resolve_project_path(
        config["outputs"]["root_directory"], config["_project_root"]
    )
    aggregate_dir = output_root / "aggregate_comparison"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.concat(score_frames, ignore_index=True)
    paired, summary = aggregate_paired_scores(scores, config)
    p_values = summarize_backtest_p_values(scores)
    scores.to_csv(aggregate_dir / "all_portfolio_risk_scores.csv", index=False)
    paired.to_csv(aggregate_dir / "paired_portfolio_scores.csv", index=False)
    summary.to_csv(aggregate_dir / "primary_scorecard_summary.csv", index=False)
    p_values.to_csv(aggregate_dir / "backtest_p_value_summary.csv", index=False)
    if config["outputs"].get("save_primary_scorecard_figures", True):
        save_scorecard_figures(summary, aggregate_dir)
        save_p_value_figures(scores, aggregate_dir)
    return {
        "all_scores": aggregate_dir / "all_portfolio_risk_scores.csv",
        "paired_scores": aggregate_dir / "paired_portfolio_scores.csv",
        "scorecard": aggregate_dir / "primary_scorecard_summary.csv",
        "p_values": aggregate_dir / "backtest_p_value_summary.csv",
    }


def execute_pipeline(config_path, dry_run=False, portfolio_limit=None):
    config = load_pipeline_config(config_path)
    plan = build_execution_plan(config)
    if portfolio_limit is not None:
        if portfolio_limit < 1:
            raise ValueError("portfolio_limit must be positive.")
        plan["portfolio_limit"] = portfolio_limit
    if dry_run:
        if config["run_scope"] in {"phase_1", "full"}:
            phase_1_plan = run_phase_1_production(config, dry_run=True)
            plan["selected_dynamic_candidate"] = phase_1_plan.get("winner")
        elif config["run_scope"] == "later_phases":
            registry = validate_selected_registry(config)
            plan["selected_dynamic_candidate"] = str(
                registry.loc[registry["role"].eq("dynamic"), "model_id"].iloc[0]
            )
        return plan

    started = datetime.now(timezone.utc)
    phase_1_result = None
    if config["run_scope"] in {"phase_1", "full"}:
        phase_1_result = run_phase_1_production(config)
    if config["run_scope"] == "phase_1":
        return {
            "status": "complete",
            "run_scope": "phase_1",
            "registry_path": phase_1_result.get("registry_path"),
            "selected_dynamic_model": phase_1_result.get("winner"),
        }

    registry = (
        phase_1_result["registry"]
        if phase_1_result is not None
        else validate_selected_registry(config)
    )
    manifests = create_or_load_portfolio_manifests(config, write=True)
    if portfolio_limit is not None:
        manifests = manifests[:portfolio_limit]

    from pipeline_wgan import run_portfolio_pipeline

    results, failures = [], []
    output_root = resolve_project_path(
        config["outputs"]["root_directory"], config["_project_root"]
    )
    for manifest in manifests:
        sample_id = int(manifest["SAMPLE_ID"].iloc[0])
        try:
            results.append(run_portfolio_pipeline(config, registry, manifest))
        except Exception as error:
            failure = {
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _write_json(output_root / f"portfolio_{sample_id:02d}" / "FAILURE.json", failure)
            if not config["execution"].get("continue_after_portfolio_failure", True):
                raise

    aggregate = _aggregate_completed_outputs(config, results)
    summary = {
        "status": "complete" if not failures else "complete_with_failures",
        "run_scope": config["run_scope"],
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "selected_dynamic_model": str(
            registry.loc[registry["role"].eq("dynamic"), "model_id"].iloc[0]
        ),
        "completed_portfolios": [result["sample_id"] for result in results],
        "failed_portfolios": [failure["sample_id"] for failure in failures],
        "aggregate_outputs": aggregate,
    }
    _write_json(output_root / "run_summary.json", summary)
    return summary


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="pipeline_config.json", help="Path to the JSON run configuration."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the plan without training."
    )
    parser.add_argument(
        "--portfolio-limit", type=int, default=None,
        help="Optional smoke-run limit; does not alter saved manifests."
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    outcome = execute_pipeline(
        arguments.config,
        dry_run=arguments.dry_run,
        portfolio_limit=arguments.portfolio_limit,
    )
    print(json.dumps(outcome, indent=2, default=str))
