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


def _completed_portfolio_results(config):
    """Discover successful portfolio outputs for a standalone aggregation job."""
    output_root = resolve_project_path(
        config["outputs"]["root_directory"], config["_project_root"]
    )
    results = []
    for success_path in sorted(output_root.glob("portfolio_*/SUCCESS.json")):
        output_dir = success_path.parent
        scores_path = output_dir / "risk_scores.csv"
        if not scores_path.exists():
            continue
        try:
            sample_id = int(output_dir.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        results.append({"sample_id": sample_id, "scores_path": scores_path})
    return results


def execute_pipeline(
    config_path,
    dry_run=False,
    portfolio_limit=None,
    portfolio_id=None,
    prepare_only=False,
    aggregate_only=False,
    run_scope=None,
):
    config = load_pipeline_config(config_path)
    if run_scope is not None:
        if run_scope not in config["allowed_run_scopes"]:
            raise ValueError(f"Unsupported run scope: {run_scope}")
        config["run_scope"] = run_scope
    plan = build_execution_plan(config)
    if portfolio_id is not None and portfolio_limit is not None:
        raise ValueError("Use either portfolio_id or portfolio_limit, not both.")
    if portfolio_id is not None and portfolio_id < 1:
        raise ValueError("portfolio_id must be positive.")
    if portfolio_limit is not None:
        if portfolio_limit < 1:
            raise ValueError("portfolio_limit must be positive.")
        plan["portfolio_limit"] = portfolio_limit
    if aggregate_only:
        results = _completed_portfolio_results(config)
        if not results:
            raise FileNotFoundError("No completed portfolio risk scores were found.")
        expected_ids = set(range(
            1, int(config["portfolio_sampling"]["number_of_samples"]) + 1
        ))
        completed_ids = {int(result["sample_id"]) for result in results}
        if completed_ids != expected_ids:
            missing = sorted(expected_ids - completed_ids)
            unexpected = sorted(completed_ids - expected_ids)
            raise RuntimeError(
                "Aggregation requires the complete configured portfolio set; "
                f"missing={missing}, unexpected={unexpected}."
            )
        aggregate = _aggregate_completed_outputs(config, results)
        summary = {
            "status": "complete",
            "run_scope": "aggregate_only",
            "completed_portfolios": [result["sample_id"] for result in results],
            "aggregate_outputs": aggregate,
        }
        output_root = resolve_project_path(
            config["outputs"]["root_directory"], config["_project_root"]
        )
        _write_json(output_root / "run_summary.json", summary)
        return summary

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
    if prepare_only:
        return {
            "status": "complete",
            "run_scope": "prepare_only",
            "selected_dynamic_model": str(
                registry.loc[registry["role"].eq("dynamic"), "model_id"].iloc[0]
            ),
            "prepared_portfolios": [int(frame["SAMPLE_ID"].iloc[0]) for frame in manifests],
        }
    if portfolio_id is not None:
        manifests = [
            frame for frame in manifests
            if int(frame["SAMPLE_ID"].iloc[0]) == portfolio_id
        ]
        if not manifests:
            raise ValueError(f"Portfolio {portfolio_id} is not present in the manifests.")
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

    if portfolio_id is not None and failures:
        failure = failures[0]
        raise RuntimeError(
            f"Portfolio {portfolio_id} failed with {failure['error_type']}: "
            f"{failure['error']}"
        )

    # A portfolio-specific process is intended for a SLURM array. Aggregation
    # is deferred to one dependency job so array tasks never write the same files.
    aggregate = None if portfolio_id is not None else _aggregate_completed_outputs(config, results)
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
    if portfolio_id is None:
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
    parser.add_argument(
        "--portfolio-id", type=int, default=None,
        help="Run exactly one saved portfolio (intended for a SLURM array task)."
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Run required Phase 1 work and create manifests, but do not train WGANs."
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Aggregate all completed portfolio outputs without training models."
    )
    parser.add_argument(
        "--run-scope", choices=["phase_1", "later_phases", "full"], default=None,
        help="Override run_scope from the JSON configuration for this invocation."
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    outcome = execute_pipeline(
        arguments.config,
        dry_run=arguments.dry_run,
        portfolio_limit=arguments.portfolio_limit,
        portfolio_id=arguments.portfolio_id,
        prepare_only=arguments.prepare_only,
        aggregate_only=arguments.aggregate_only,
        run_scope=arguments.run_scope,
    )
    print(json.dumps(outcome, indent=2, default=str))
