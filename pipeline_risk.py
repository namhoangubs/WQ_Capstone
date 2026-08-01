"""Risk forecast scoring and paired portfolio-level model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


EPSILON = np.finfo(float).eps


def _xlogy(count, probability):
    if count == 0:
        return 0.0
    return float(count) * np.log(np.clip(probability, EPSILON, 1.0 - EPSILON))


def kupiec_unconditional_coverage(exceedances, expected_probability):
    """Return the likelihood-ratio statistic and p-value for VaR coverage."""
    hits = np.asarray(exceedances, dtype=bool)
    n_obs = int(hits.size)
    if n_obs == 0:
        return np.nan, np.nan
    n_hits = int(hits.sum())
    observed = n_hits / n_obs
    null_log_likelihood = _xlogy(n_hits, expected_probability) + _xlogy(
        n_obs - n_hits, 1.0 - expected_probability
    )
    alternative_log_likelihood = _xlogy(n_hits, observed) + _xlogy(
        n_obs - n_hits, 1.0 - observed
    )
    statistic = max(0.0, 2.0 * (alternative_log_likelihood - null_log_likelihood))
    return statistic, float(chi2.sf(statistic, 1))


def christoffersen_independence(exceedances):
    """Test whether VaR exceedances cluster rather than arrive independently."""
    hits = np.asarray(exceedances, dtype=int)
    if hits.size < 2:
        return np.nan, np.nan
    previous, current = hits[:-1], hits[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else 0.0
    pi_01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi_11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    null_log_likelihood = _xlogy(n01 + n11, pi) + _xlogy(n00 + n10, 1.0 - pi)
    alternative_log_likelihood = (
        _xlogy(n01, pi_01)
        + _xlogy(n00, 1.0 - pi_01)
        + _xlogy(n11, pi_11)
        + _xlogy(n10, 1.0 - pi_11)
    )
    statistic = max(0.0, 2.0 * (alternative_log_likelihood - null_log_likelihood))
    return statistic, float(chi2.sf(statistic, 1))


def christoffersen_conditional_coverage(exceedances, expected_probability):
    coverage_statistic, _ = kupiec_unconditional_coverage(
        exceedances, expected_probability
    )
    independence_statistic, _ = christoffersen_independence(exceedances)
    if not np.isfinite(coverage_statistic) or not np.isfinite(independence_statistic):
        return np.nan, np.nan
    statistic = coverage_statistic + independence_statistic
    return statistic, float(chi2.sf(statistic, 2))


def quantile_loss(realized, var_forecast, tail_probability):
    realized = np.asarray(realized, dtype=float)
    forecast = np.asarray(var_forecast, dtype=float)
    valid = np.isfinite(realized) & np.isfinite(forecast)
    if not valid.any():
        return np.nan
    residual = realized[valid] - forecast[valid]
    return float(np.mean((tail_probability - (residual < 0)) * residual))


def fissler_ziegel_zero_loss(realized, var_forecast, es_forecast, tail_probability):
    """Mean FZ0 joint VaR-ES loss for lower-tail return forecasts."""
    realized = np.asarray(realized, dtype=float)
    var = np.asarray(var_forecast, dtype=float)
    es = np.asarray(es_forecast, dtype=float)
    valid = np.isfinite(realized) & np.isfinite(var) & np.isfinite(es) & (es < 0)
    if not valid.any():
        return np.nan
    realized, var, es = realized[valid], var[valid], es[valid]
    hits = realized <= var
    score = (
        -(hits * (var - realized)) / (tail_probability * es)
        + var / es
        + np.log(-es)
        - 1.0
    )
    return float(np.mean(score))


def _level_suffix(confidence):
    return f"{100.0 * float(confidence):g}".replace(".", "")


def score_risk_forecasts(forecasts, config, sample_id):
    """Create tidy whole-period and stress-period scores for one portfolio."""
    frame = forecasts.copy()
    frame["DATE"] = pd.to_datetime(frame["DATE"])
    rows = []
    periods = [{"name": "full_forecast", "start": None, "end": None}]
    periods.extend(config["evaluation"].get("stress_periods", []))
    primary_levels = set(config["evaluation"]["primary_scorecard_levels"])

    for model_id, model_frame in frame.groupby("MODEL_ID"):
        for period in periods:
            sample = model_frame
            if period["start"]:
                sample = sample[sample["DATE"] >= pd.Timestamp(period["start"])]
            if period["end"]:
                sample = sample[sample["DATE"] <= pd.Timestamp(period["end"])]
            for confidence in config["evaluation"]["original_risk_levels"]:
                suffix = _level_suffix(confidence)
                realized = sample["REALIZED_RETURN"].to_numpy(dtype=float)
                var = sample[f"VAR_{suffix}"].to_numpy(dtype=float)
                es = sample[f"ES_{suffix}"].to_numpy(dtype=float)
                valid = np.isfinite(realized) & np.isfinite(var) & np.isfinite(es)
                realized, var, es = realized[valid], var[valid], es[valid]
                tail_probability = 1.0 - float(confidence)
                hits = realized < var
                hit_rate = float(hits.mean()) if hits.size else np.nan
                uc_lr, uc_p = kupiec_unconditional_coverage(hits, tail_probability)
                ind_lr, ind_p = christoffersen_independence(hits)
                cc_lr, cc_p = christoffersen_conditional_coverage(
                    hits, tail_probability
                )
                empirical_es = float(realized[hits].mean()) if hits.any() else np.nan
                metrics = {
                    "var_exceedance_rate": hit_rate,
                    "var_exceedance_rate_error": abs(hit_rate - tail_probability),
                    "kupiec_lr": uc_lr,
                    "kupiec_p_value": uc_p,
                    "christoffersen_independence_lr": ind_lr,
                    "christoffersen_independence_p_value": ind_p,
                    "christoffersen_conditional_coverage_lr": cc_lr,
                    "christoffersen_conditional_coverage_p_value": cc_p,
                    "var_quantile_loss": quantile_loss(realized, var, tail_probability),
                    "fissler_ziegel_joint_var_es_loss": fissler_ziegel_zero_loss(
                        realized, var, es, tail_probability
                    ),
                }
                if period["name"] != "full_forecast":
                    metrics["stress_period_tail_risk"] = (
                        abs(empirical_es - float(np.mean(es)))
                        if np.isfinite(empirical_es) else np.nan
                    )
                for metric, value in metrics.items():
                    rows.append({
                        "SAMPLE_ID": sample_id,
                        "MODEL_ID": model_id,
                        "PERIOD": period["name"],
                        "CONFIDENCE_LEVEL": float(confidence),
                        "PRIMARY_LEVEL": float(confidence) in primary_levels,
                        "METRIC": metric,
                        "VALUE": value,
                        "N_OBSERVATIONS": int(hits.size),
                        "N_EXCEEDANCES": int(hits.sum()),
                    })
    return pd.DataFrame(rows)


def rolling_exceedance_counts(forecasts, window=250):
    """Retain the original paper's rolling 250-day VaR exceedance diagnostic."""
    rows = []
    for model_id, frame in forecasts.groupby("MODEL_ID"):
        frame = frame.sort_values("DATE")
        for confidence in (0.975, 0.99):
            suffix = _level_suffix(confidence)
            hits = (frame["REALIZED_RETURN"] < frame[f"VAR_{suffix}"]).astype(int)
            counts = hits.rolling(window, min_periods=window).sum()
            for date, value in zip(frame["DATE"], counts):
                if np.isfinite(value):
                    rows.append({
                        "DATE": date,
                        "MODEL_ID": model_id,
                        "CONFIDENCE_LEVEL": confidence,
                        "WINDOW": window,
                        "EXCEEDANCE_COUNT": int(value),
                    })
    return pd.DataFrame(rows)


def aggregate_paired_scores(scores, config):
    """Summarize dynamic-minus-fixed score differences across portfolios."""
    performance_metrics = {
        "var_exceedance_rate_error",
        "var_quantile_loss",
        "fissler_ziegel_joint_var_es_loss",
        "stress_period_tail_risk",
    }
    scores = scores[scores["METRIC"].isin(performance_metrics)].copy()
    dynamic_ids = [
        value for value in scores["MODEL_ID"].unique()
        if value not in {"fixed_hmm_transition", "normal"}
    ]
    if len(dynamic_ids) != 1:
        raise ValueError("Paired aggregation requires exactly one dynamic model.")
    dynamic_id = dynamic_ids[0]
    fixed = scores[scores["MODEL_ID"].eq("fixed_hmm_transition")]
    dynamic = scores[scores["MODEL_ID"].eq(dynamic_id)]
    keys = ["SAMPLE_ID", "PERIOD", "CONFIDENCE_LEVEL", "METRIC"]
    paired = fixed.merge(dynamic, on=keys, suffixes=("_FIXED", "_DYNAMIC"))
    paired["DYNAMIC_MINUS_FIXED"] = paired["VALUE_DYNAMIC"] - paired["VALUE_FIXED"]

    settings = config["evaluation"]["portfolio_aggregation"]
    replicates = int(settings["bootstrap_replicates"])
    confidence = float(settings["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(int(config["portfolio_sampling"]["random_seed"]))
    rows = []
    group_keys = ["PERIOD", "CONFIDENCE_LEVEL", "METRIC"]
    for values, group in paired.groupby(group_keys):
        differences = group["DYNAMIC_MINUS_FIXED"].dropna().to_numpy(dtype=float)
        if differences.size:
            bootstrap = rng.choice(
                differences, size=(replicates, differences.size), replace=True
            ).mean(axis=1)
            lower, upper = np.quantile(bootstrap, [alpha, 1.0 - alpha])
        else:
            lower = upper = np.nan
        rows.append({
            "DYNAMIC_MODEL_ID": dynamic_id,
            "PERIOD": values[0],
            "CONFIDENCE_LEVEL": values[1],
            "METRIC": values[2],
            "N_PORTFOLIOS": int(differences.size),
            "FIXED_MEAN": group["VALUE_FIXED"].mean(),
            "DYNAMIC_MEAN": group["VALUE_DYNAMIC"].mean(),
            "MEAN_DIFFERENCE_DYNAMIC_MINUS_FIXED": np.mean(differences) if differences.size else np.nan,
            "MEDIAN_DIFFERENCE_DYNAMIC_MINUS_FIXED": np.median(differences) if differences.size else np.nan,
            "BOOTSTRAP_CI_LOWER": lower,
            "BOOTSTRAP_CI_UPPER": upper,
            "DYNAMIC_WIN_RATE": np.mean(differences < 0) if differences.size else np.nan,
        })
    return paired, pd.DataFrame(rows)


def summarize_backtest_p_values(scores, significance_level=0.05):
    p_values = scores[scores["METRIC"].str.endswith("_p_value")].dropna(
        subset=["VALUE"]
    )
    rows = []
    keys = ["MODEL_ID", "PERIOD", "CONFIDENCE_LEVEL", "METRIC"]
    for values, frame in p_values.groupby(keys):
        rows.append({
            "MODEL_ID": values[0],
            "PERIOD": values[1],
            "CONFIDENCE_LEVEL": values[2],
            "METRIC": values[3],
            "N_PORTFOLIOS": len(frame),
            "MEAN_P_VALUE": frame["VALUE"].mean(),
            "MEDIAN_P_VALUE": frame["VALUE"].median(),
            "P_VALUE_Q25": frame["VALUE"].quantile(0.25),
            "P_VALUE_Q75": frame["VALUE"].quantile(0.75),
            "REJECTION_RATE_AT_5_PERCENT": np.mean(
                frame["VALUE"] < significance_level
            ),
        })
    return pd.DataFrame(rows)


def save_p_value_figures(scores, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = scores[
        scores["METRIC"].str.endswith("_p_value")
        & scores["PERIOD"].eq("full_forecast")
        & scores["PRIMARY_LEVEL"]
    ].dropna(subset=["VALUE"])
    for metric, frame in sample.groupby("METRIC"):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for model_id, model_frame in frame.groupby("MODEL_ID"):
            axis.hist(
                model_frame["VALUE"], bins=np.linspace(0, 1, 11), alpha=0.45,
                label=model_id
            )
        axis.axvline(0.05, color="red", linestyle="--", linewidth=0.8)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xlabel("Backtest p-value")
        axis.set_ylabel("Portfolio count")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}_distribution.png", dpi=180)
        plt.close(figure)


def save_scorecard_figures(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for confidence in sorted(summary["CONFIDENCE_LEVEL"].unique()):
        sample = summary[
            summary["CONFIDENCE_LEVEL"].eq(confidence)
            & summary["PERIOD"].eq("full_forecast")
        ].dropna(subset=[
            "MEAN_DIFFERENCE_DYNAMIC_MINUS_FIXED",
            "BOOTSTRAP_CI_LOWER",
            "BOOTSTRAP_CI_UPPER",
        ])
        if sample.empty:
            continue
        figure, axis = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(sample))
        means = sample["MEAN_DIFFERENCE_DYNAMIC_MINUS_FIXED"].to_numpy()
        lower = np.maximum(
            0.0, means - sample["BOOTSTRAP_CI_LOWER"].to_numpy()
        )
        upper = np.maximum(
            0.0, sample["BOOTSTRAP_CI_UPPER"].to_numpy() - means
        )
        axis.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=5)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, sample["METRIC"], rotation=20, ha="right")
        axis.set_ylabel("Dynamic minus fixed (lower is better)")
        axis.set_title(f"Paired portfolio score differences at {confidence:.1%}")
        figure.tight_layout()
        figure.savefig(output_dir / f"paired_scorecard_{_level_suffix(confidence)}.png", dpi=180)
        plt.close(figure)


def save_portfolio_risk_figures(forecasts, rolling, output_dir):
    """Save counterparts of the original paper's VaR and exceedance plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts = forecasts.copy()
    forecasts["DATE"] = pd.to_datetime(forecasts["DATE"])
    for confidence, limit in [(0.975, 30), (0.99, 12)]:
        sample = rolling[rolling["CONFIDENCE_LEVEL"].eq(confidence)]
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for model_id, frame in sample.groupby("MODEL_ID"):
            axis.plot(frame["DATE"], frame["EXCEEDANCE_COUNT"], linewidth=0.8, label=model_id)
        axis.axhline(limit, color="red", linestyle="--", linewidth=0.8, label="Paper limit")
        axis.set_title(f"Rolling 250-day {confidence:.1%} VaR exceedances")
        axis.set_ylabel("Exceedance count")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"rolling_exceedances_{_level_suffix(confidence)}.png", dpi=180)
        plt.close(figure)

    for model_id, frame in forecasts.groupby("MODEL_ID"):
        frame = frame.sort_values("DATE")
        figure, axis = plt.subplots(figsize=(10, 4.5))
        axis.plot(frame["DATE"], frame["REALIZED_RETURN"], linewidth=0.45, color="black", label="Realized")
        for confidence in (0.9, 0.95, 0.99):
            suffix = _level_suffix(confidence)
            axis.plot(frame["DATE"], frame[f"VAR_{suffix}"], linewidth=0.65, label=f"{confidence:.1%} VaR")
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set_title(f"One-step VaR forecasts: {model_id}")
        axis.set_ylabel("Equal-weight portfolio return")
        axis.legend(fontsize=8, ncol=4)
        figure.tight_layout()
        figure.savefig(output_dir / f"var_forecasts_{model_id}.png", dpi=180)
        plt.close(figure)
