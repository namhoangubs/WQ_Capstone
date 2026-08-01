"""Production q4 HMM refit and fixed/dynamic transition handoff."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

from p1_3_transition_features import build_transition_dataset
from p1_4_dynamic_transition import _align_probabilities, fit_dynamic_transition_model
from p1_4_mlg_transition import (
    MLGFeatureTransformer,
    MLGTransitionModel,
    PreparedMLGDesign,
    best_mlg_config,
    load_or_create_mlg_report,
)
from p1_5_transition_comparison import load_or_create_dynamic_report
from p1_10_weighted_mlg_transition import (
    GroupWeightedMLGTransitionModel,
    best_weighted_mlg_config,
    load_or_create_weighted_mlg_report,
)
from p1_11_pvalue_vif_mlr_transition import (
    PValueVIFMLRModel,
    SelectionResult,
    best_pvalue_vif_mlr_config,
    build_corrected_transition_dataset,
    load_or_create_pvalue_vif_mlr_report,
)
from p1_12_state_interaction_mlg_transition import (
    PreparedStateInteractionDesign,
    StateInteractionMLGTransformer,
    StateInteractionMLGTransitionModel,
    best_state_interaction_mlg_config,
    duration_free_feature_frame,
    load_or_create_state_interaction_mlg_report,
)
from pipeline_runtime import (
    PipelineConfigError,
    load_model_specifications,
    resolve_project_path,
    selected_registry_path,
    validate_selected_registry,
)
from project_config import BASE_DIR


MODEL_LOADERS = {
    "dynamic_duration_logit": load_or_create_dynamic_report,
    "pvalue_vif_selected_mlr": load_or_create_pvalue_vif_mlr_report,
    "multinomial_logistic_gam": load_or_create_mlg_report,
    "group_weighted_multinomial_logistic_gam": load_or_create_weighted_mlg_report,
    "state_interaction_multinomial_logistic_gam": load_or_create_state_interaction_mlg_report,
}


def _pipe_list(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    return [item for item in str(value).split("|") if item and item.lower() != "nan"]


def _json_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return {}
    text = str(value).strip()
    return {} if not text or text.lower() == "nan" else json.loads(text)


def _relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if not isinstance(value, (dict, list, tuple)) and pd.isna(value):
        return None
    return value


def _best_original_mlr(report):
    candidates = report[
        report["MODEL"].eq("dynamic_duration_logit")
        & report["SPLIT"].eq("validation")
    ].copy()
    if candidates.empty:
        raise PipelineConfigError("Original MLR report has no validation candidates.")
    return candidates.sort_values(
        ["LOG_LOSS", "BRIER_SCORE", "STATE_LAG", "EXOG_LAG"],
        ascending=[True, True, True, True],
    ).iloc[0]


def _best_row(model_id, report):
    if model_id == "dynamic_duration_logit":
        return _best_original_mlr(report)
    if model_id == "pvalue_vif_selected_mlr":
        return best_pvalue_vif_mlr_config(report)
    if model_id == "multinomial_logistic_gam":
        return best_mlg_config(report)
    if model_id == "group_weighted_multinomial_logistic_gam":
        return best_weighted_mlg_config(report)
    if model_id == "state_interaction_multinomial_logistic_gam":
        return best_state_interaction_mlg_config(report)
    raise PipelineConfigError(f"Unsupported dynamic model: {model_id}")


def _complexity(row):
    for field in ["SELECTED_TERM_COUNT", "SELECTED_FEATURE_COUNT"]:
        if field in row and pd.notna(row[field]):
            return int(row[field])
    return 1 + int(row.get("STATE_LAG", 0)) + int(row.get("EXOG_LAG", 0))


def select_best_dynamic_candidate(config, create_missing=True):
    """Select across model families using validation data only."""
    specifications = load_model_specifications(config)
    specifications = specifications[
        specifications["enabled"] & specifications["eligible_for_full_selection"]
    ]
    candidates = []
    for specification in specifications.itertuples(index=False):
        report_path = resolve_project_path(
            specification.selection_report_path, config["_project_root"]
        )
        if report_path.exists():
            report = pd.read_csv(report_path)
        elif create_missing:
            report = MODEL_LOADERS[specification.model_id](4)
        else:
            raise PipelineConfigError(f"Phase 1 report is missing: {report_path}")
        best = _best_row(specification.model_id, report)
        candidates.append({
            "model_id": specification.model_id,
            "display_name": specification.display_name,
            "log_loss": float(best["LOG_LOSS"]),
            "brier_score": float(best["BRIER_SCORE"]),
            "accuracy": float(best["ACCURACY"]),
            "complexity": _complexity(best),
            "best_row": best,
            "report_path": report_path,
        })

    selection = config["phase_1"]["cross_model_selection"]
    minimum = min(candidate["log_loss"] for candidate in candidates)
    tolerance = float(selection["tie_tolerance"])
    finalists = [
        candidate for candidate in candidates
        if candidate["log_loss"] <= minimum + tolerance
    ]
    finalists.sort(
        key=lambda candidate: (
            candidate["complexity"],
            candidate["brier_score"],
            -candidate["accuracy"],
            candidate["model_id"],
        )
    )
    winner = finalists[0]
    finalist_ids = {candidate["model_id"] for candidate in finalists}
    winner_id = winner["model_id"]
    summary = pd.DataFrame([
        {
            "MODEL_ID": candidate["model_id"],
            "DISPLAY_NAME": candidate["display_name"],
            "VALIDATION_LOG_LOSS": candidate["log_loss"],
            "VALIDATION_BRIER_SCORE": candidate["brier_score"],
            "VALIDATION_ACCURACY": candidate["accuracy"],
            "MODEL_COMPLEXITY": candidate["complexity"],
            "WITHIN_LOG_LOSS_TOLERANCE": candidate["model_id"] in finalist_ids,
            "SELECTED": candidate["model_id"] == winner_id,
        }
        for candidate in candidates
    ]).sort_values(["VALIDATION_LOG_LOSS", "MODEL_COMPLEXITY"])
    return winner, summary


def _canonical_state_mapping(model, exog, n_states):
    """Match refitted raw HMM labels to the established Phase 1 state labels."""
    reference_path = BASE_DIR / "phase_1" / f"stock_rets_q{n_states}.csv"
    if not reference_path.exists():
        return {state: state for state in range(n_states)}
    reference_states = pd.read_csv(
        reference_path, usecols=["DATE", "STATE"], index_col="DATE", parse_dates=True
    )["STATE"]
    reference = exog.join(reference_states, how="inner").groupby("STATE").mean()
    reference = reference.reindex(range(n_states))
    if reference.isna().any(axis=None):
        raise PipelineConfigError(
            "Established q4 state assignments do not cover every canonical state."
        )
    canonical = reference.to_numpy(dtype=float)
    raw = np.asarray(model.means_, dtype=float)
    scale = np.nanstd(canonical, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    cost = np.linalg.norm((raw[:, None, :] - canonical[None, :, :]) / scale, axis=2)
    raw_rows, canonical_columns = linear_sum_assignment(cost)
    return {
        int(raw_state): int(canonical_state)
        for raw_state, canonical_state in zip(raw_rows, canonical_columns)
    }


def _mapped_transition_matrix(raw_matrix, mapping, n_states):
    matrix = np.zeros((n_states, n_states), dtype=float)
    for raw_origin in range(n_states):
        for raw_destination in range(n_states):
            matrix[mapping[raw_origin], mapping[raw_destination]] = raw_matrix[
                raw_origin, raw_destination
            ]
    return matrix / matrix.sum(axis=1, keepdims=True)


def _fit_stable_production_hmm(calibration, n_states, config, output_dir):
    """Select the best deterministic restart that passes production stability gates."""
    settings = config["phase_1"]["production_hmm_refit"]
    old_params = pd.read_csv(BASE_DIR / "phase_1" / "best_params.csv")
    original_seed = int(
        old_params.loc[old_params["N_STATES"].eq(n_states), "SEED"].iloc[0]
    )
    restart_count = int(settings["random_restarts"])
    rng = np.random.default_rng(int(settings["random_seed"]))
    seeds = [original_seed]
    while len(seeds) < restart_count:
        candidate = int(rng.integers(0, 1_000_000))
        if candidate not in seeds:
            seeds.append(candidate)

    reports, eligible = [], []
    minimum_count = int(settings["minimum_regime_observations"])
    maximum_condition = float(settings["maximum_covariance_condition_number"])
    maximum_decrease = float(settings["maximum_final_log_likelihood_decrease"])
    for seed in seeds:
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                random_state=seed,
                n_iter=10000,
                tol=1e-5,
                implementation="scaling",
            ).fit(calibration)
            labels = model.predict(calibration)
            counts = np.bincount(labels, minlength=n_states)
            conditions = np.array(
                [np.linalg.cond(covariance) for covariance in model.covars_]
            )
            history = list(model.monitor_.history)
            final_delta = history[-1] - history[-2] if len(history) > 1 else np.nan
            score = float(model.score(calibration))
            converged_before_limit = int(model.monitor_.iter) < 10000
            stable = (
                counts.min() >= minimum_count
                and np.isfinite(conditions).all()
                and conditions.max() <= maximum_condition
                and converged_before_limit
                and (not np.isfinite(final_delta) or final_delta >= -maximum_decrease)
            )
            report = {
                "SEED": seed,
                "LOG_LIKELIHOOD": score,
                "ITERATIONS": int(model.monitor_.iter),
                "CONVERGED_BEFORE_MAX_ITERATIONS": converged_before_limit,
                "MIN_REGIME_COUNT": int(counts.min()),
                "MAX_COVARIANCE_CONDITION_NUMBER": float(conditions.max()),
                "FINAL_LOG_LIKELIHOOD_DELTA": final_delta,
                "PASSES_STABILITY_GATES": stable,
                "ERROR": "",
            }
            reports.append(report)
            if stable:
                eligible.append((score, seed, model))
        except Exception as error:
            reports.append({
                "SEED": seed,
                "LOG_LIKELIHOOD": np.nan,
                "ITERATIONS": 0,
                "CONVERGED_BEFORE_MAX_ITERATIONS": False,
                "MIN_REGIME_COUNT": 0,
                "MAX_COVARIANCE_CONDITION_NUMBER": np.nan,
                "FINAL_LOG_LIKELIHOOD_DELTA": np.nan,
                "PASSES_STABILITY_GATES": False,
                "ERROR": f"{type(error).__name__}: {error}",
            })

    report_path = Path(output_dir) / f"production_hmm_refit_candidates_q{n_states}.csv"
    pd.DataFrame(reports).sort_values(
        ["PASSES_STABILITY_GATES", "LOG_LIKELIHOOD"], ascending=[False, False]
    ).to_csv(report_path, index=False)
    if not eligible:
        raise PipelineConfigError(
            f"No q{n_states} HMM restart passed the production stability gates; see {report_path}."
        )
    score, seed, model = max(eligible, key=lambda item: (item[0], -item[1]))
    return model, seed, report_path


def fit_production_hmm(config, output_dir):
    """Refit q4 through 2016 and infer post-2016 states without future observations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_states = int(config["phase_1"]["n_regimes"])
    validation_end = pd.Timestamp(config["data"]["validation_end_exclusive"])
    window_size = int(config["wgan"]["rolling_window_size"])

    exog = pd.read_csv(
        BASE_DIR / "processed" / "exog_rets.csv", index_col=0, parse_dates=[0]
    ).dropna()
    calibration = exog[exog.index < validation_end]
    model, seed, restart_report_path = _fit_stable_production_hmm(
        calibration, n_states, config, output_dir
    )
    mapping = _canonical_state_mapping(model, calibration, n_states)

    raw_states = pd.Series(index=exog.index, dtype=int, name="STATE_RAW")
    raw_states.loc[calibration.index] = model.predict(calibration)
    for date in exog.index[exog.index >= validation_end]:
        window = exog.loc[:date].tail(window_size)
        raw_states.loc[date] = int(model.predict(window)[-1])
    raw_states = raw_states.astype(int)
    states = raw_states.map(mapping).astype(int).rename("STATE")

    hmm_path = output_dir / f"production_hmm_q{n_states}.joblib"
    mapping_path = output_dir / f"production_hmm_state_mapping_q{n_states}.json"
    matrix_path = output_dir / f"production_fixed_transition_q{n_states}.csv"
    states_path = output_dir / f"production_states_q{n_states}.csv"
    stock_path = output_dir / f"production_stock_rets_q{n_states}.csv"
    joblib.dump(model, hmm_path)
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "raw_to_canonical": {str(key): value for key, value in mapping.items()},
            "seed": seed,
            "calibration_start": str(calibration.index.min().date()),
            "calibration_end": str(calibration.index.max().date()),
            "exog_columns": list(exog.columns),
        }, handle, indent=2)
    matrix = _mapped_transition_matrix(model.transmat_, mapping, n_states)
    pd.DataFrame(matrix).to_csv(matrix_path)
    pd.concat([raw_states, states], axis=1).to_csv(states_path)

    stock = pd.read_csv(
        BASE_DIR / "processed" / "stock_rets.csv", index_col=0, parse_dates=[0]
    )
    stock = stock.join(states, how="inner")
    stock.to_csv(stock_path)
    return {
        "model": model,
        "mapping": mapping,
        "states": states,
        "exog": exog,
        "fixed_matrix": matrix,
        "hmm_path": hmm_path,
        "mapping_path": mapping_path,
        "matrix_path": matrix_path,
        "states_path": states_path,
        "stock_path": stock_path,
        "restart_report_path": restart_report_path,
    }


def _transition_dates(states, exog, feature_index):
    aligned_dates = pd.concat([states.rename("STATE"), exog], axis=1).dropna().index
    mapping = pd.Series(aligned_dates[1:], index=aligned_dates[:-1])
    return pd.to_datetime(mapping.reindex(feature_index))


def _calibration_prediction_masks(config, states, exog, X):
    forecast_dates = _transition_dates(states, exog, X.index)
    validation_end = pd.Timestamp(config["data"]["validation_end_exclusive"])
    forecast_start = pd.Timestamp(config["data"]["forecast_start"])
    calibration = forecast_dates < validation_end
    prediction = forecast_dates >= forecast_start
    return calibration.fillna(False), prediction.fillna(False), forecast_dates


def _prepared_mlg(X, y, best):
    transformer = MLGFeatureTransformer(
        n_knots=int(best["N_KNOTS"]), degree=3
    ).fit(X)
    selected_terms = _pipe_list(best["SELECTED_TERMS"])
    missing = set(selected_terms) - set(transformer.term_names_)
    if missing:
        raise PipelineConfigError(f"Selected MLG terms unavailable after refit: {sorted(missing)}")
    return PreparedMLGDesign(
        transformer=transformer,
        selected_terms=selected_terms,
        term_p_values=_json_dict(best.get("TERM_P_VALUES", "{}")),
        X_train=X,
        y_train=y,
        inference_method="frozen_phase_1_specification_refit",
    )


def fit_selected_dynamic_on_calibration(config, winner, states, exog):
    """Refit only coefficients/preprocessing; never repeat model selection."""
    model_id = winner["model_id"]
    best = winner["best_row"]
    state_lag = int(best.get("STATE_LAG", 0))
    exog_lag = int(best.get("EXOG_LAG", 0))

    if model_id == "pvalue_vif_selected_mlr":
        X, y, meta, classes, term_columns = build_corrected_transition_dataset(
            states,
            exog,
            max_state_lag=state_lag,
            max_exog_lag=exog_lag,
            reference_state=int(best.get("REFERENCE_STATE", 0)),
        )
    else:
        X, y, meta, classes = build_transition_dataset(
            states, exog, max_state_lag=state_lag, max_exog_lag=exog_lag
        )
        term_columns = None
        if model_id == "state_interaction_multinomial_logistic_gam":
            X = duration_free_feature_frame(X)

    calibration, prediction, forecast_dates = _calibration_prediction_masks(
        config, states, exog, X
    )
    X_cal, y_cal = X.loc[calibration], y.loc[calibration]
    if y_cal.nunique() != len(classes):
        raise PipelineConfigError("Production transition calibration does not contain every q4 state.")

    if model_id == "dynamic_duration_logit":
        model = fit_dynamic_transition_model(X_cal, y_cal)
    elif model_id == "pvalue_vif_selected_mlr":
        selected_terms = _pipe_list(best["SELECTED_TERMS"])
        selected_columns = [
            column for term in selected_terms for column in term_columns[term]
        ]
        selection = SelectionResult(
            selected_terms=selected_terms,
            selected_columns=selected_columns,
            term_p_values=_json_dict(best.get("TERM_P_VALUES", "{}")),
            term_vifs=_json_dict(best.get("TERM_VIFS", "{}")),
            feature_vifs=_json_dict(best.get("FEATURE_VIFS", "{}")),
            history=[],
            scaler=StandardScaler().fit(X_cal[selected_columns]),
        )
        model = PValueVIFMLRModel(selection, X_cal, y_cal)
    elif model_id == "multinomial_logistic_gam":
        model = MLGTransitionModel(
            _prepared_mlg(X_cal, y_cal, best), float(best["REGULARIZATION_C"])
        )
    elif model_id == "group_weighted_multinomial_logistic_gam":
        prepared = _prepared_mlg(X_cal, y_cal, best)
        model = GroupWeightedMLGTransitionModel(
            prepared,
            float(best["REGULARIZATION_C"]),
            {
                "state": float(best["STATE_PENALTY_WEIGHT"]),
                "market": float(best["MARKET_PENALTY_WEIGHT"]),
                "duration": float(best["DURATION_PENALTY_WEIGHT"]),
            },
        )
    elif model_id == "state_interaction_multinomial_logistic_gam":
        base_transformer = MLGFeatureTransformer(
            n_knots=int(best["N_KNOTS"]), degree=3
        ).fit(X_cal)
        base_terms = _pipe_list(best["SELECTED_BASE_TERMS"])
        interaction_map = _json_dict(best.get("INTERACTION_STATE_MAP", "{}"))
        transformer = StateInteractionMLGTransformer(
            base_transformer, base_terms, interaction_map
        )
        selected_interactions = _pipe_list(best.get("SELECTED_INTERACTIONS", ""))
        prepared = PreparedStateInteractionDesign(
            transformer=transformer,
            base_terms=base_terms,
            selected_interactions=selected_interactions,
            base_p_values={term: np.nan for term in base_terms},
            interaction_p_values=_json_dict(best.get("INTERACTION_P_VALUES", "{}")),
            interaction_fdr_p_values=_json_dict(best.get("INTERACTION_FDR_P_VALUES", "{}")),
            interaction_methods=_json_dict(best.get("INTERACTION_METHODS", "{}")),
            interaction_improvements=_json_dict(best.get("INTERACTION_BOOTSTRAP_IMPROVEMENTS", "{}")),
            interaction_replicates=_json_dict(best.get("INTERACTION_BOOTSTRAP_VALID_REPLICATES", "{}")),
            X_train=X_cal,
            y_train=y_cal,
        )
        model = StateInteractionMLGTransitionModel(
            prepared, float(best["REGULARIZATION_C"])
        )
    else:
        raise PipelineConfigError(f"Unsupported selected model: {model_id}")

    X_prediction = X.loc[prediction]
    probabilities = _align_probabilities(
        model.predict_proba(X_prediction), model.classes_, classes
    )
    return {
        "model": model,
        "model_id": model_id,
        "best": best,
        "X_calibration": X_cal,
        "y_calibration": y_cal,
        "X_prediction": X_prediction,
        "meta_prediction": meta.loc[prediction],
        "forecast_dates": forecast_dates.loc[prediction],
        "probabilities": probabilities,
        "classes": classes,
        "state_lag": state_lag,
        "exog_lag": exog_lag,
    }


def _classifier_and_names(dynamic):
    model = dynamic["model"]
    X = dynamic["X_calibration"]
    if hasattr(model, "pipeline"):
        classifier = model.pipeline.named_steps["logit"]
        names = list(model.selected_columns)
        terms = names
    elif hasattr(model, "named_steps"):
        classifier = model.named_steps["logit"]
        names = list(X.columns)
        terms = names
    else:
        classifier = model.classifier
        slices = model.transformer.term_slices(X, model.selected_terms)
        names, terms = [], []
        for term, term_slice in slices.items():
            for basis in range(term_slice.stop - term_slice.start):
                names.append(f"{term}__basis_{basis}")
                terms.append(term)
    return classifier, names, terms


def save_dynamic_artifacts(config, hmm, dynamic, output_dir):
    output_dir = Path(output_dir)
    model_id = dynamic["model_id"]
    model_path = output_dir / f"{model_id}_production_q4.joblib"
    coefficients_path = output_dir / f"{model_id}_coefficients_q4.csv"
    preprocessing_path = output_dir / f"{model_id}_preprocessing_q4.json"
    spline_path = output_dir / f"{model_id}_spline_q4.joblib"
    probability_path = output_dir / "production_transition_probabilities_q4.csv"
    joblib.dump(dynamic["model"], model_path)

    classifier, names, terms = _classifier_and_names(dynamic)
    coefficient_matrix = getattr(
        dynamic["model"], "effective_coefficients_", classifier.coef_
    )
    rows = []
    for class_index, class_label in enumerate(classifier.classes_):
        coefficient_index = min(class_index, coefficient_matrix.shape[0] - 1)
        rows.append({
            "CLASS": int(class_label),
            "TERM": "INTERCEPT",
            "TRANSFORMED_FEATURE": "INTERCEPT",
            "COEFFICIENT": float(classifier.intercept_[coefficient_index]),
        })
        for feature, term, coefficient in zip(
            names, terms, coefficient_matrix[coefficient_index]
        ):
            rows.append({
                "CLASS": int(class_label),
                "TERM": term,
                "TRANSFORMED_FEATURE": feature,
                "COEFFICIENT": float(coefficient),
            })
    pd.DataFrame(rows).to_csv(coefficients_path, index=False)

    preprocessing = {
        "model_id": model_id,
        "classes": [int(value) for value in dynamic["classes"]],
        "state_lag": dynamic["state_lag"],
        "exog_lag": dynamic["exog_lag"],
        "raw_feature_columns": list(dynamic["X_calibration"].columns),
        "selected_terms": list(getattr(dynamic["model"], "selected_terms", names)),
        "calibration_start": str(dynamic["X_calibration"].index.min().date()),
        "calibration_end": str(dynamic["X_calibration"].index.max().date()),
        "selection_row": {
            str(key): _json_safe(value) for key, value in dynamic["best"].to_dict().items()
        },
    }
    with preprocessing_path.open("w", encoding="utf-8") as handle:
        json.dump(preprocessing, handle, indent=2)
    if hasattr(dynamic["model"], "transformer"):
        joblib.dump(dynamic["model"].transformer, spline_path)
        spline_value = _relative(spline_path)
    else:
        spline_value = ""

    current_states = dynamic["meta_prediction"]["CURRENT_STATE"].to_numpy(dtype=int)
    fixed = hmm["fixed_matrix"][current_states]
    frame = pd.DataFrame({
        "FORECAST_DATE": dynamic["forecast_dates"].to_numpy(),
        "INFORMATION_DATE": dynamic["X_prediction"].index,
        "CURRENT_STATE": current_states,
        "DYNAMIC_MODEL_ID": model_id,
    })
    for column, state in enumerate(dynamic["classes"]):
        frame[f"FIXED_P_TO_{state}"] = fixed[:, column]
        frame[f"DYNAMIC_P_TO_{state}"] = dynamic["probabilities"][:, column]
    frame["FIXED_ROW_SUM"] = fixed.sum(axis=1)
    frame["DYNAMIC_ROW_SUM"] = dynamic["probabilities"].sum(axis=1)
    frame.to_csv(probability_path, index=False)
    return {
        "model_path": model_path,
        "coefficients_path": coefficients_path,
        "preprocessing_path": preprocessing_path,
        "spline_path": spline_value,
        "probability_path": probability_path,
    }


def write_selected_registry(config, hmm, dynamic, artifacts, output_dir):
    path = selected_registry_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "n_states": 4,
        "hmm_artifact": _relative(hmm["hmm_path"]),
        "fixed_transition_matrix_artifact": _relative(hmm["matrix_path"]),
        "transition_probability_file": _relative(artifacts["probability_path"]),
        "production_stock_returns": _relative(hmm["stock_path"]),
        "refit_end_exclusive": config["data"]["validation_end_exclusive"],
    }
    rows = [
        {
            **common,
            "role": "baseline",
            "model_id": "fixed_hmm_transition",
            "model_family": "fixed_hmm",
            "selection_mode": "fixed",
            "fitted_model_artifact": _relative(hmm["hmm_path"]),
            "coefficient_file": "",
            "preprocessing_file": _relative(hmm["mapping_path"]),
            "spline_artifact": "",
        },
        {
            **common,
            "role": "dynamic",
            "model_id": dynamic["model_id"],
            "model_family": dynamic["model_id"],
            "selection_mode": "auto_best_validation",
            "fitted_model_artifact": _relative(artifacts["model_path"]),
            "coefficient_file": _relative(artifacts["coefficients_path"]),
            "preprocessing_file": _relative(artifacts["preprocessing_path"]),
            "spline_artifact": artifacts["spline_path"],
        },
    ]
    registry = pd.DataFrame(rows)
    registry.to_csv(path, index=False)
    return path, registry


def run_phase_1_production(config, dry_run=False):
    """Run/validate Phase 1 and produce the strict later-phase handoff."""
    policy = config["phase_1"]["full_run_model_policy"]
    if policy["dynamic_model_selection"] == "manual_frozen":
        return {"registry": validate_selected_registry(config), "registry_path": selected_registry_path(config)}

    winner, selection_summary = select_best_dynamic_candidate(
        config, create_missing=not dry_run
    )
    if dry_run:
        return {"winner": winner["model_id"], "selection_summary": selection_summary}

    output_dir = BASE_DIR / "phase_1" / "production_q4"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_summary.to_csv(output_dir / "cross_model_validation_selection_q4.csv", index=False)
    hmm = fit_production_hmm(config, output_dir)
    dynamic = fit_selected_dynamic_on_calibration(
        config, winner, hmm["states"], hmm["exog"]
    )
    artifacts = save_dynamic_artifacts(config, hmm, dynamic, output_dir)
    registry_path, registry = write_selected_registry(
        config, hmm, dynamic, artifacts, output_dir
    )
    return {
        "winner": winner["model_id"],
        "selection_summary": selection_summary,
        "hmm": hmm,
        "dynamic": dynamic,
        "artifacts": artifacts,
        "registry_path": registry_path,
        "registry": registry,
    }
