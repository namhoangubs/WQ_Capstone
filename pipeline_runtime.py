"""Configuration, validation, and reproducible portfolio manifests."""

from __future__ import annotations

import json
from itertools import combinations
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
    _require(
        sampling.get("sampling_design") in {
            "independent_random",
            "balanced_low_overlap",
        },
        "portfolio_sampling.sampling_design must be independent_random or balanced_low_overlap.",
    )
    _require(
        int(sampling.get("balanced_construction_attempts", 0)) > 0,
        "portfolio_sampling.balanced_construction_attempts must be positive.",
    )
    _require(
        int(sampling.get("pairwise_swap_iterations", 0)) >= 0,
        "portfolio_sampling.pairwise_swap_iterations cannot be negative.",
    )

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
        int(wgan.get("initial_training_iterations", 0)) > 0,
        "wgan.initial_training_iterations must be positive.",
    )
    _require(
        int(wgan.get("rolling_training_iterations", 0)) > 0,
        "wgan.rolling_training_iterations must be positive.",
    )
    _require(
        int(wgan["rolling_training_iterations"])
        <= int(wgan["initial_training_iterations"]),
        "Rolling WGAN iterations cannot exceed initial WGAN iterations.",
    )
    _require(
        int(config["execution"].get("portfolio_workers", 0)) == 1,
        "This deterministic TensorFlow runner currently requires portfolio_workers=1.",
    )
    _require(
        int(config["execution"].get("progress_interval_dates", 0)) > 0,
        "execution.progress_interval_dates must be positive.",
    )
    _require(
        int(config["execution"].get("memory_cleanup_interval_dates", 0)) > 0,
        "execution.memory_cleanup_interval_dates must be positive.",
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


def _sampling_summary_path(directory):
    return Path(directory) / "portfolio_sampling_summary.json"


def _balanced_column_targets(total, column_count, rng):
    """Spread a fixed number of assignments as evenly as possible over columns."""
    base, remainder = divmod(int(total), int(column_count))
    targets = np.full(column_count, base, dtype=int)
    if remainder:
        targets[rng.permutation(column_count)[:remainder]] += 1
    return targets


def _pairwise_overlap_matrix(assignment):
    """Return portfolio-by-portfolio ticker overlap counts."""
    return np.asarray(assignment, dtype=int).T @ np.asarray(assignment, dtype=int)


def _pairwise_overlap_objective(assignment):
    overlaps = _pairwise_overlap_matrix(assignment)
    upper = overlaps[np.triu_indices_from(overlaps, k=1)].astype(float)
    if upper.size == 0:
        return 0.0
    return float(np.square(upper - upper.mean()).sum())


def _allocate_replication_group(
    assignment,
    stock_indexes,
    replication_count,
    column_targets,
    pair_overlaps,
    rng,
):
    """Allocate one replication group while preserving exact portfolio capacities."""
    if replication_count == 0:
        return np.array(column_targets, dtype=int)

    remaining = np.asarray(column_targets, dtype=int).copy()
    for stock_index in rng.permutation(stock_indexes):
        available = np.flatnonzero(remaining > 0)
        if available.size < replication_count:
            return None
        candidates = list(combinations(available, replication_count))
        capacity_sums = np.array(
            [remaining[list(candidate)].sum() for candidate in candidates], dtype=int
        )
        # Capacity is the primary feasibility criterion. Pair overlap is the
        # secondary criterion that spreads shared stocks across portfolio pairs.
        highest_capacity = capacity_sums.max()
        capacity_candidates = [
            candidate
            for candidate, capacity in zip(candidates, capacity_sums)
            if capacity == highest_capacity
        ]
        pair_scores = np.array(
            [
                sum(pair_overlaps[left, right] for left, right in combinations(candidate, 2))
                for candidate in capacity_candidates
            ],
            dtype=float,
        )
        best_candidates = [
            candidate
            for candidate, score in zip(capacity_candidates, pair_scores)
            if score == pair_scores.min()
        ]
        selected = best_candidates[int(rng.integers(len(best_candidates)))]
        assignment[stock_index, list(selected)] = 1
        remaining[list(selected)] -= 1
        for left, right in combinations(selected, 2):
            pair_overlaps[left, right] += 1
            pair_overlaps[right, left] += 1
    return remaining if not remaining.any() else None


def _improve_pairwise_overlap(assignment, rng, iterations):
    """Use incidence-preserving swaps to flatten pairwise portfolio overlap."""
    assignment = np.asarray(assignment, dtype=np.int8)
    portfolio_count = assignment.shape[1]
    if portfolio_count < 2 or iterations == 0:
        return assignment
    portfolio_pairs = list(combinations(range(portfolio_count), 2))
    objective = _pairwise_overlap_objective(assignment)
    for _ in range(int(iterations)):
        left, right = portfolio_pairs[int(rng.integers(len(portfolio_pairs)))]
        left_only = np.flatnonzero(
            (assignment[:, left] == 1) & (assignment[:, right] == 0)
        )
        right_only = np.flatnonzero(
            (assignment[:, right] == 1) & (assignment[:, left] == 0)
        )
        if not left_only.size or not right_only.size:
            continue
        left_stock = int(left_only[rng.integers(left_only.size)])
        right_stock = int(right_only[rng.integers(right_only.size)])
        candidate = assignment.copy()
        candidate[left_stock, left], candidate[left_stock, right] = 0, 1
        candidate[right_stock, right], candidate[right_stock, left] = 0, 1
        candidate_objective = _pairwise_overlap_objective(candidate)
        if candidate_objective < objective:
            assignment = candidate
            objective = candidate_objective
    return assignment


def _balanced_low_overlap_assignment(
    universe,
    sample_count,
    stocks_per_sample,
    rng,
    construction_attempts,
    pairwise_swap_iterations,
):
    """Construct a reproducible low-overlap assignment with balanced inclusions."""
    universe_size = len(universe)
    total_assignments = int(sample_count) * int(stocks_per_sample)
    minimum_replications, extra_replications = divmod(total_assignments, universe_size)
    replication_counts = np.full(universe_size, minimum_replications, dtype=int)
    if extra_replications:
        replication_counts[rng.permutation(universe_size)[:extra_replications]] += 1

    best_assignment = None
    best_objective = np.inf
    for _ in range(int(construction_attempts)):
        assignment = np.zeros((universe_size, sample_count), dtype=np.int8)
        pair_overlaps = np.zeros((sample_count, sample_count), dtype=int)
        success = True
        # Allocate the higher-replication group first. Its portfolio targets are
        # balanced before the lower-replication group fills the residual slots.
        high_replication = minimum_replications + 1
        high_indexes = np.flatnonzero(replication_counts == high_replication)
        high_targets = _balanced_column_targets(
            len(high_indexes) * high_replication, sample_count, rng
        )
        if high_targets.max(initial=0) > stocks_per_sample:
            success = False
        elif _allocate_replication_group(
            assignment,
            high_indexes,
            high_replication,
            high_targets,
            pair_overlaps,
            rng,
        ) is None:
            success = False

        low_indexes = np.flatnonzero(replication_counts == minimum_replications)
        low_targets = np.repeat(stocks_per_sample, sample_count) - high_targets
        if success and _allocate_replication_group(
            assignment,
            low_indexes,
            minimum_replications,
            low_targets,
            pair_overlaps,
            rng,
        ) is None:
            success = False
        if not success:
            continue
        if not np.array_equal(assignment.sum(axis=1), replication_counts):
            continue
        if not np.all(assignment.sum(axis=0) == stocks_per_sample):
            continue

        assignment = _improve_pairwise_overlap(
            assignment, rng, pairwise_swap_iterations
        )
        objective = _pairwise_overlap_objective(assignment)
        if objective < best_objective:
            best_assignment = assignment
            best_objective = objective

    if best_assignment is None:
        raise PipelineConfigError(
            "Could not construct balanced low-overlap portfolios. Increase "
            "balanced_construction_attempts or revise the portfolio dimensions."
        )
    return best_assignment, replication_counts


def _sampling_diagnostics(universe, assignment, sampling):
    overlaps = _pairwise_overlap_matrix(assignment)
    pairwise = overlaps[np.triu_indices_from(overlaps, k=1)]
    frequencies = assignment.sum(axis=1).astype(int)
    return {
        "sampling_design": sampling["sampling_design"],
        "random_seed": int(sampling["random_seed"]),
        "eligible_universe_size": int(len(universe)),
        "portfolio_count": int(assignment.shape[1]),
        "stocks_per_portfolio": int(assignment.shape[0] and assignment.sum(axis=0)[0]),
        "total_assignments": int(assignment.sum()),
        "minimum_stock_inclusion_count": int(frequencies.min()),
        "maximum_stock_inclusion_count": int(frequencies.max()),
        "pairwise_overlap_minimum": int(pairwise.min()) if pairwise.size else 0,
        "pairwise_overlap_mean": float(pairwise.mean()) if pairwise.size else 0.0,
        "pairwise_overlap_maximum": int(pairwise.max()) if pairwise.size else 0,
    }


def validate_portfolio_manifest(frame, universe, stocks_per_sample):
    required = {"SAMPLE_ID", "POSITION", "TICKER", "WEIGHT", "SAMPLING_SEED"}
    _require(required.issubset(frame.columns), f"Portfolio manifest columns missing: {sorted(required - set(frame.columns))}")
    _require(len(frame) == int(stocks_per_sample), "Portfolio manifest has the wrong number of stocks.")
    _require(frame["TICKER"].is_unique, "Portfolio manifest contains duplicate tickers.")
    _require(set(frame["TICKER"]).issubset(set(universe)), "Portfolio manifest contains ineligible tickers.")
    _require(np.isclose(frame["WEIGHT"].sum(), 1.0), "Portfolio weights must sum to one.")
    return frame.sort_values("POSITION").reset_index(drop=True)


def create_or_load_portfolio_manifests(config, write=True):
    """Create deterministic portfolio manifests and persist exact membership."""
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
    if write:
        directory.mkdir(parents=True, exist_ok=True)
    manifest_paths = [_manifest_path(directory, number) for number in range(1, sample_count + 1)]
    summary_path = _sampling_summary_path(directory)
    reusable = sampling.get("reuse_existing_manifests", True) and all(
        path.exists() for path in manifest_paths
    )
    if reusable and sampling["sampling_design"] == "balanced_low_overlap":
        _require(
            summary_path.exists(),
            "Balanced low-overlap manifests require portfolio_sampling_summary.json. "
            "Set reuse_existing_manifests=false to regenerate legacy manifests.",
        )
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        _require(
            summary.get("sampling_design") == "balanced_low_overlap",
            "Existing manifests were not created with balanced_low_overlap. "
            "Set reuse_existing_manifests=false to regenerate them.",
        )
    if reusable:
        return [
            validate_portfolio_manifest(pd.read_csv(path), universe, stocks_per_sample)
            for path in manifest_paths
        ]

    if sampling["sampling_design"] == "independent_random":
        assignment = np.zeros((len(universe), sample_count), dtype=np.int8)
        for sample_number in range(sample_count):
            selected = rng.choice(len(universe), size=stocks_per_sample, replace=False)
            assignment[selected, sample_number] = 1
    else:
        assignment, _ = _balanced_low_overlap_assignment(
            universe,
            sample_count,
            stocks_per_sample,
            rng,
            sampling["balanced_construction_attempts"],
            sampling["pairwise_swap_iterations"],
        )

    manifests = []
    for sample_number in range(1, sample_count + 1):
        selected = np.flatnonzero(assignment[:, sample_number - 1])
        tickers = np.asarray(universe, dtype=object)[selected]
        frame = pd.DataFrame({
            "SAMPLE_ID": sample_number,
            "POSITION": np.arange(1, stocks_per_sample + 1),
            "TICKER": tickers,
            "WEIGHT": np.repeat(1.0 / stocks_per_sample, stocks_per_sample),
            "SAMPLING_SEED": int(sampling["random_seed"]),
            "SAMPLING_DESIGN": sampling["sampling_design"],
        })
        if write:
            frame.to_csv(manifest_paths[sample_number - 1], index=False)
        manifests.append(validate_portfolio_manifest(frame, universe, stocks_per_sample))

    if write:
        pd.concat(manifests, ignore_index=True).to_csv(
            directory / "portfolio_manifest_all.csv", index=False
        )
        overlap_labels = [f"PORTFOLIO_{number:02d}" for number in range(1, sample_count + 1)]
        overlap = pd.DataFrame(
            _pairwise_overlap_matrix(assignment),
            index=overlap_labels,
            columns=overlap_labels,
        )
        overlap.index.name = "PORTFOLIO"
        overlap.to_csv(directory / "portfolio_overlap_matrix.csv")
        pd.DataFrame({
            "TICKER": universe,
            "PORTFOLIO_INCLUSION_COUNT": assignment.sum(axis=1).astype(int),
        }).to_csv(directory / "portfolio_stock_inclusion_frequency.csv", index=False)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(_sampling_diagnostics(universe, assignment, sampling), handle, indent=2)
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
