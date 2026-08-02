"""Configuration, validation, and reproducible portfolio manifests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import BASE_DIR, INPUT_DIR


class PipelineConfigError(ValueError):
    """Raised when a pipeline configuration is incomplete or inconsistent."""


def _require(condition, message):
    if not condition:
        raise PipelineConfigError(message)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_project_path(value, project_root=BASE_DIR):
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def load_pipeline_config(path=BASE_DIR / "pipeline_config.json"):
    """Load and validate the human-editable JSON run contract."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    allowed_scopes = set(config.get("allowed_run_scopes", []))
    _require(
        allowed_scopes == {"phase_1", "later_phases", "full"},
        "allowed_run_scopes must contain phase_1, later_phases, and full.",
    )
    _require(
        config.get("run_scope") in allowed_scopes,
        f"Unknown run_scope: {config.get('run_scope')}",
    )
    _require(
        config.get("data", {}).get("source") == "original_paper",
        "This pipeline version requires data.source='original_paper'.",
    )
    _require(
        not config["data"].get("allow_post_2023_extension", False),
        "Post-2023 extension is disabled by the agreed design.",
    )

    phase_1 = config.get("phase_1", {})
    _require(int(phase_1.get("n_regimes", 0)) == 4, "Only the q4 HMM is supported.")
    hmm_refit = phase_1.get("production_hmm_refit", {})
    _require(
        int(hmm_refit.get("random_restarts", 0)) > 0,
        "production_hmm_refit.random_restarts must be positive.",
    )
    _require(
        int(hmm_refit.get("minimum_regime_observations", 0)) > 0,
        "production_hmm_refit.minimum_regime_observations must be positive.",
    )
    policy = phase_1.get("full_run_model_policy", {})
    allowed_selection = set(policy.get("allowed_dynamic_model_selection", []))
    _require(
        policy.get("dynamic_model_selection") in allowed_selection,
        "dynamic_model_selection is not listed in allowed_dynamic_model_selection.",
    )
    if policy.get("dynamic_model_selection") == "manual_frozen":
        _require(
            bool(policy.get("manual_dynamic_model_id")),
            "manual_dynamic_model_id is required for manual_frozen mode.",
        )

    sampling = config.get("portfolio_sampling", {})
    _require(int(sampling.get("number_of_samples", 0)) > 0, "number_of_samples must be positive.")
    _require(int(sampling.get("stocks_per_sample", 0)) > 0, "stocks_per_sample must be positive.")
    _require(
        sampling.get("without_replacement_within_sample") is True,
        "Stocks must be sampled without replacement within each portfolio.",
    )
    _require(sampling.get("weighting") == "equal", "Only equal portfolio weights are supported.")

    simulation = config.get("simulation", {})
    _require(
        int(simulation.get("paths_per_forecast_date", 0)) > 0,
        "paths_per_forecast_date must be positive.",
    )
    _require(
        simulation.get("paired_random_numbers_across_models") is True,
        "Paired random numbers are required for model comparisons.",
    )
    wgan = config.get("wgan", {})
    _require(
        wgan.get("initial_train_end_exclusive")
        == config["data"]["initial_train_end_exclusive"],
        "The HMM-WGAN initial training cutoffs must match.",
    )
    _require(
        wgan.get("warmup_end_exclusive")
        == config["data"]["validation_end_exclusive"],
        "The WGAN warmup and Phase 1 validation cutoffs must match.",
    )
    _require(
        wgan.get("share_generators_across_transition_models") is True,
        "Fixed and dynamic transitions must share the same WGAN generators.",
    )
    _require(
        wgan.get("training_mode") == "adaptive_regime_memory_ess_gated",
        "wgan.training_mode must be 'adaptive_regime_memory_ess_gated'.",
    )
    memory = wgan.get("regime_memory", {})
    _require(
        memory.get("weighting") == "exponential",
        "wgan.regime_memory.weighting must be 'exponential'.",
    )
    _require(
        float(memory.get("recency_half_life_trading_days", 0)) > 0,
        "wgan.regime_memory.recency_half_life_trading_days must be positive.",
    )
    _require(
        float(memory.get("minimum_effective_sample_size", 0)) == 64,
        "wgan.regime_memory.minimum_effective_sample_size must be 64.",
    )
    _require(
        int(wgan.get("rolling_window_size", 0)) > 0,
        "wgan.rolling_window_size must be positive.",
    )
    _require(
        int(wgan.get("batch_size", 0)) <= 64,
        "wgan.batch_size cannot exceed the ESS update threshold of 64.",
    )
    _require(
        int(config["execution"].get("portfolio_workers", 0)) == 1,
        "This deterministic TensorFlow runner currently requires portfolio_workers=1.",
    )
    _require(
        config["outputs"].get("save_all_simulated_paths") is False,
        "Saving every daily Monte Carlo path is intentionally disabled for this experiment.",
    )
    _require(
        pd.Timestamp(config["data"]["forecast_start"])
        >= pd.Timestamp(config["data"]["validation_end_exclusive"]),
        "forecast_start cannot precede the validation endpoint.",
    )

    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent)
    return config


def load_model_specifications(config):
    """Read the flat Phase 1 candidate registry and parse each JSON specification."""
    project_root = Path(config["_project_root"])
    path = resolve_project_path(
        config["phase_1"]["model_specifications_csv"], project_root
    )
    _require(path.exists(), f"Model specification file does not exist: {path}")
    rows = pd.read_csv(path)
    required = {
        "enabled",
        "eligible_for_full_selection",
        "model_id",
        "role",
        "specification_mode",
        "specification_json",
        "selection_report_path",
    }
    _require(required.issubset(rows.columns), f"Model specification columns missing: {sorted(required - set(rows.columns))}")
    _require(rows["model_id"].is_unique, "model_id values must be unique.")
    rows["enabled"] = rows["enabled"].map(_as_bool)
    rows["eligible_for_full_selection"] = rows["eligible_for_full_selection"].map(_as_bool)
    parsed = []
    for row in rows.itertuples(index=False):
        try:
            parsed.append(json.loads(row.specification_json))
        except json.JSONDecodeError as error:
            raise PipelineConfigError(
                f"Invalid specification_json for {row.model_id}: {error}"
            ) from error
    rows["parsed_specification"] = parsed

    baselines = rows[rows["enabled"] & rows["role"].eq("baseline")]
    _require(len(baselines) == 1, "Exactly one enabled baseline model is required.")
    dynamic = rows[rows["enabled"] & rows["eligible_for_full_selection"]]
    _require(not dynamic.empty, "At least one enabled dynamic candidate is required.")
    return rows


def eligible_stock_universe(input_dir=INPUT_DIR):
    """Return stocks active in 1999 and 2023 with complete original-paper prices."""
    input_dir = Path(input_dir)
    status_1999 = pd.read_csv(input_dir / "STOCK_STATUS_1999.csv")
    status_2023 = pd.read_csv(input_dir / "STOCK_STATUS_2023.csv")
    active_1999 = set(status_1999.loc[status_1999["STATUS"].eq("Active"), "TICKER"])
    active_2023 = set(status_2023.loc[status_2023["STATUS"].eq("Active"), "TICKER"])
    candidates = sorted(active_1999 & active_2023)

    prices = pd.read_csv(input_dir / "STOCKS.csv", index_col=0, parse_dates=[0])
    candidates = [ticker for ticker in candidates if ticker in prices.columns]
    complete = prices[candidates].dropna(axis=1, how="any").columns.tolist()
    return sorted(complete)


def _manifest_path(directory, sample_number):
    return Path(directory) / f"portfolio_{sample_number:02d}.csv"


def validate_portfolio_manifest(frame, universe, stocks_per_sample):
    required = {"SAMPLE_ID", "POSITION", "TICKER", "WEIGHT", "SAMPLING_SEED"}
    _require(required.issubset(frame.columns), f"Portfolio manifest columns missing: {sorted(required - set(frame.columns))}")
    _require(len(frame) == int(stocks_per_sample), "Portfolio manifest has the wrong number of stocks.")
    _require(frame["TICKER"].is_unique, "Portfolio manifest contains duplicate tickers.")
    _require(set(frame["TICKER"]).issubset(set(universe)), "Portfolio manifest contains ineligible tickers.")
    _require(np.isclose(frame["WEIGHT"].sum(), 1.0), "Portfolio weights must sum to one.")
    return frame.sort_values("POSITION").reset_index(drop=True)


def create_or_load_portfolio_manifests(config, write=True):
    """Create deterministic independent draws and persist their exact ticker membership."""
    sampling = config["portfolio_sampling"]
    project_root = Path(config["_project_root"])
    directory = resolve_project_path(sampling["manifest_directory"], project_root)
    universe = eligible_stock_universe()
    sample_count = int(sampling["number_of_samples"])
    stocks_per_sample = int(sampling["stocks_per_sample"])
    _require(
        stocks_per_sample <= len(universe),
        f"Requested {stocks_per_sample} stocks but only {len(universe)} are eligible.",
    )

    rng = np.random.default_rng(int(sampling["random_seed"]))
    manifests = []
    if write:
        directory.mkdir(parents=True, exist_ok=True)
    for sample_number in range(1, sample_count + 1):
        path = _manifest_path(directory, sample_number)
        if sampling.get("reuse_existing_manifests", True) and path.exists():
            frame = pd.read_csv(path)
        else:
            tickers = rng.choice(universe, size=stocks_per_sample, replace=False)
            frame = pd.DataFrame({
                "SAMPLE_ID": sample_number,
                "POSITION": np.arange(1, stocks_per_sample + 1),
                "TICKER": tickers,
                "WEIGHT": np.repeat(1.0 / stocks_per_sample, stocks_per_sample),
                "SAMPLING_SEED": int(sampling["random_seed"]),
            })
            if write:
                frame.to_csv(path, index=False)
        manifests.append(validate_portfolio_manifest(frame, universe, stocks_per_sample))

    if write:
        pd.concat(manifests, ignore_index=True).to_csv(
            directory / "portfolio_manifest_all.csv", index=False
        )
    return manifests


def selected_registry_path(config):
    project_root = Path(config["_project_root"])
    policy = config["phase_1"]["full_run_model_policy"]
    if policy["dynamic_model_selection"] == "manual_frozen":
        value = policy["manual_frozen_model"]["frozen_model_registry"]
    else:
        value = config["phase_1"]["later_phase_artifacts"]["selected_model_registry"]
    return resolve_project_path(value, project_root)


def validate_selected_registry(config, path=None):
    """Validate the fixed-plus-one-dynamic handoff consumed by later phases."""
    path = selected_registry_path(config) if path is None else Path(path)
    _require(path.exists(), f"Selected-model registry is missing: {path}")
    registry = pd.read_csv(path).fillna("")
    required_columns = {
        "role",
        "model_id",
        "n_states",
        "hmm_artifact",
        "fixed_transition_matrix_artifact",
        "transition_probability_file",
        "production_stock_returns",
        "fitted_model_artifact",
        "coefficient_file",
        "preprocessing_file",
        "spline_artifact",
    }
    _require(required_columns.issubset(registry.columns), f"Selected registry columns missing: {sorted(required_columns - set(registry.columns))}")
    _require(len(registry[registry["role"].eq("baseline")]) == 1, "Registry must contain one baseline row.")
    dynamic = registry[registry["role"].eq("dynamic")]
    _require(len(dynamic) == 1, "Registry must contain exactly one dynamic row.")
    _require(registry["n_states"].astype(int).eq(4).all(), "Registry must contain q4 artifacts only.")

    family = str(dynamic.iloc[0].get("model_family", "")).lower()
    dynamic_artifacts = [
        "fitted_model_artifact",
        "coefficient_file",
        "preprocessing_file",
    ]
    if "gam" in family or "mlg" in family:
        dynamic_artifacts.append("spline_artifact")

    if config["phase_1"]["full_run_model_policy"]["dynamic_model_selection"] == "manual_frozen":
        expected = config["phase_1"]["full_run_model_policy"]["manual_dynamic_model_id"]
        _require(dynamic.iloc[0]["model_id"] == expected, "Frozen registry model_id does not match manual_dynamic_model_id.")
        manual = config["phase_1"]["full_run_model_policy"]["manual_frozen_model"]
        required_artifacts = list(manual["required_artifacts"])
        if "gam" in family or "mlg" in family:
            required_artifacts += list(manual["required_artifacts_for_gam_models"])
        for artifact in required_artifacts:
            _require(artifact in registry.columns, f"Frozen registry is missing {artifact}.")
            value = dynamic.iloc[0][artifact]
            _require(bool(value), f"Frozen registry has no value for {artifact}.")
            _require(
                resolve_project_path(value, config["_project_root"]).exists(),
                f"Frozen registry artifact is missing: {value}",
            )

    for artifact in dynamic_artifacts:
        value = dynamic.iloc[0][artifact]
        _require(bool(value), f"Dynamic registry has no value for {artifact}.")
        _require(
            resolve_project_path(value, config["_project_root"]).exists(),
            f"Dynamic registry artifact is missing: {value}",
        )

    for column in [
        "hmm_artifact",
        "fixed_transition_matrix_artifact",
        "transition_probability_file",
        "production_stock_returns",
    ]:
        for value in registry[column].unique():
            _require(resolve_project_path(value, config["_project_root"]).exists(), f"Registry artifact is missing: {value}")

    probability_path = resolve_project_path(
        dynamic.iloc[0]["transition_probability_file"], config["_project_root"]
    )
    probability_frame = pd.read_csv(probability_path)
    probability_columns = {"FORECAST_DATE", "CURRENT_STATE"}
    probability_columns.update(f"FIXED_P_TO_{state}" for state in range(4))
    probability_columns.update(f"DYNAMIC_P_TO_{state}" for state in range(4))
    _require(
        probability_columns.issubset(probability_frame.columns),
        f"Transition probability columns missing: {sorted(probability_columns - set(probability_frame.columns))}",
    )
    _require(
        not probability_frame["FORECAST_DATE"].duplicated().any(),
        "Transition probability file contains duplicate forecast dates.",
    )
    for prefix in ["FIXED", "DYNAMIC"]:
        columns = [f"{prefix}_P_TO_{state}" for state in range(4)]
        values = probability_frame[columns].to_numpy(dtype=float)
        _require(np.isfinite(values).all(), f"{prefix} probabilities contain non-finite values.")
        _require((values >= 0).all(), f"{prefix} probabilities contain negative values.")
        _require(
            np.allclose(values.sum(axis=1), 1.0, atol=1e-8),
            f"{prefix} transition probabilities do not sum to one.",
        )
    return registry


def build_execution_plan(config):
    """Return a side-effect-free summary used by dry runs and tests."""
    specs = load_model_specifications(config)
    manifests = []
    if config["run_scope"] in {"later_phases", "full"}:
        manifests = create_or_load_portfolio_manifests(config, write=False)
    return {
        "run_scope": config["run_scope"],
        "n_regimes": int(config["phase_1"]["n_regimes"]),
        "hmm_random_restarts": int(
            config["phase_1"]["production_hmm_refit"]["random_restarts"]
        ),
        "enabled_phase_1_models": specs.loc[specs["enabled"], "model_id"].tolist(),
        "eligible_dynamic_models": specs.loc[
            specs["enabled"] & specs["eligible_for_full_selection"], "model_id"
        ].tolist(),
        "portfolio_count": len(manifests),
        "stocks_per_portfolio": int(config["portfolio_sampling"]["stocks_per_sample"]),
        "forecast_start": config["data"]["forecast_start"],
        "forecast_end": config["data"]["forecast_end"],
    }
