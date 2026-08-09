"""Causal volatility filtering for the dynamic-scale WGAN experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_runtime import resolve_project_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_config(settings):
    keys = [
        "model",
        "innovation_distribution",
        "conditional_mean",
        "fit_end_exclusive",
        "validation_fit_end_exclusive",
        "validation_end_exclusive",
        "fallback_order",
        "ewma_decay",
        "variance_floor",
        "maximum_persistence",
    ]
    return {key: settings.get(key) for key in keys}


def _cache_paths(config):
    settings = config["dynamic_scale"]
    root = resolve_project_path(settings["cache_directory"], config["_project_root"])
    return {
        "root": root,
        "sigma": root / "daily_conditional_sigma.csv.gz",
        "residuals": root / "standardized_residuals.csv.gz",
        "diagnostics": root / "volatility_model_diagnostics.csv",
        "metadata": root / "volatility_cache_metadata.json",
    }


def _registry_stock_path(config, registry):
    values = [
        value
        for value in registry["production_stock_returns"].astype(str).unique()
        if value
    ]
    if len(values) != 1:
        raise ValueError("Registry must identify one production stock-return artifact.")
    return resolve_project_path(values[0], config["_project_root"])


def _initial_variance(values, floor):
    sample = np.asarray(values[: min(len(values), 252)], dtype=float)
    variance = float(np.nanvar(sample, ddof=1)) if sample.size > 1 else floor
    return max(variance, floor)


def gjr_variance_recursion(
    returns,
    omega,
    alpha,
    gamma,
    beta,
    *,
    variance_floor=1e-10,
):
    """Return one-step-ahead variances; row t uses returns only through t-1."""
    values = np.asarray(returns, dtype=float)
    variances = np.empty(values.size, dtype=float)
    if values.size == 0:
        return variances
    persistence = float(alpha) + 0.5 * float(gamma) + float(beta)
    unconditional = (
        float(omega) / (1.0 - persistence)
        if 0.0 <= persistence < 1.0
        else variance_floor
    )
    variances[0] = max(unconditional, variance_floor)
    for position in range(1, values.size):
        previous = values[position - 1]
        shock = previous * previous
        asymmetric = shock if previous < 0 else 0.0
        variances[position] = (
            float(omega)
            + float(alpha) * shock
            + float(gamma) * asymmetric
            + float(beta) * variances[position - 1]
        )
        if not np.isfinite(variances[position]):
            raise ValueError("Conditional variance recursion produced a non-finite value.")
        variances[position] = max(variances[position], variance_floor)
    return variances


def ewma_variance_recursion(returns, decay=0.94, variance_floor=1e-10):
    values = np.asarray(returns, dtype=float)
    variances = np.empty(values.size, dtype=float)
    if values.size == 0:
        return variances
    variances[0] = _initial_variance(values, variance_floor)
    for position in range(1, values.size):
        previous = values[position - 1]
        variances[position] = (
            float(decay) * variances[position - 1]
            + (1.0 - float(decay)) * previous * previous
        )
        variances[position] = max(variances[position], variance_floor)
    return variances


def _fit_arch_parameters(returns, asymmetric, settings):
    try:
        from arch import arch_model
    except ImportError as error:
        raise ImportError(
            "Dynamic Scale requires the 'arch' package. Install requirements.txt."
        ) from error

    percentage_returns = 100.0 * np.asarray(returns, dtype=float)
    model = arch_model(
        percentage_returns,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1 if asymmetric else 0,
        q=1,
        dist="StudentsT",
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)
    if int(result.convergence_flag) != 0:
        raise ValueError(f"ARCH optimizer convergence flag: {result.convergence_flag}")
    parameters = result.params
    alpha = float(parameters["alpha[1]"])
    gamma = float(parameters.get("gamma[1]", 0.0))
    beta = float(parameters["beta[1]"])
    persistence = alpha + 0.5 * gamma + beta
    if not np.isfinite(persistence) or persistence >= float(
        settings["maximum_persistence"]
    ):
        raise ValueError(f"Invalid volatility persistence: {persistence}")
    values = {
        # ARCH was fitted to percentage returns; convert omega to decimal-return units.
        "omega": float(parameters["omega"]) / 10000.0,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "nu": float(parameters.get("nu", np.nan)),
        "persistence": persistence,
        "log_likelihood": float(result.loglikelihood),
    }
    if not np.isfinite(list(values.values())[:4]).all():
        raise ValueError("Volatility parameters contain non-finite values.")
    return values


def _fit_with_fallback(returns, settings):
    errors = []
    for method in settings["fallback_order"]:
        try:
            if method == "gjr_garch_t":
                parameters = _fit_arch_parameters(returns, True, settings)
                variance = gjr_variance_recursion(returns, **{
                    key: parameters[key]
                    for key in ["omega", "alpha", "gamma", "beta"]
                }, variance_floor=float(settings["variance_floor"]))
            elif method == "garch_t":
                parameters = _fit_arch_parameters(returns, False, settings)
                variance = gjr_variance_recursion(returns, **{
                    key: parameters[key]
                    for key in ["omega", "alpha", "gamma", "beta"]
                }, variance_floor=float(settings["variance_floor"]))
            elif method == "ewma":
                parameters = {
                    "omega": np.nan,
                    "alpha": 1.0 - float(settings["ewma_decay"]),
                    "gamma": 0.0,
                    "beta": float(settings["ewma_decay"]),
                    "nu": np.nan,
                    "persistence": 1.0,
                    "log_likelihood": np.nan,
                }
                variance = ewma_variance_recursion(
                    returns,
                    decay=float(settings["ewma_decay"]),
                    variance_floor=float(settings["variance_floor"]),
                )
            else:
                raise ValueError(f"Unknown volatility fallback method: {method}")
            return method, parameters, variance, " | ".join(errors)
        except ImportError:
            # A missing package is an environment error, not evidence that the
            # requested volatility model failed for this return series.
            raise
        except Exception as error:  # Each numerical failure is retained in diagnostics.
            errors.append(f"{method}: {type(error).__name__}: {error}")
    raise RuntimeError("All volatility models failed: " + " | ".join(errors))


def _qlike(returns, variances):
    realized = np.square(np.asarray(returns, dtype=float))
    forecast = np.asarray(variances, dtype=float)
    valid = np.isfinite(realized) & np.isfinite(forecast) & (forecast > 0)
    if not valid.any():
        return np.nan
    ratio = realized[valid] / forecast[valid]
    return float(np.mean(np.log(forecast[valid]) + ratio))


def _fit_one_stock(series, settings):
    fit_end = pd.Timestamp(settings["fit_end_exclusive"])
    validation_fit_end = pd.Timestamp(settings["validation_fit_end_exclusive"])
    validation_end = pd.Timestamp(settings["validation_end_exclusive"])
    production = series[series.index < fit_end].dropna()
    validation_fit = series[series.index < validation_fit_end].dropna()
    if len(production) < 500 or len(validation_fit) < 500:
        raise ValueError("At least 500 observations are required for volatility fitting.")

    validation_method, validation_parameters, validation_variance, validation_errors = (
        _fit_with_fallback(validation_fit.to_numpy(dtype=float), settings)
    )
    validation_full_variance = (
        ewma_variance_recursion(
            series.to_numpy(dtype=float),
            decay=float(settings["ewma_decay"]),
            variance_floor=float(settings["variance_floor"]),
        )
        if validation_method == "ewma"
        else gjr_variance_recursion(
            series.to_numpy(dtype=float),
            **{
                key: validation_parameters[key]
                for key in ["omega", "alpha", "gamma", "beta"]
            },
            variance_floor=float(settings["variance_floor"]),
        )
    )
    validation_mask = (series.index >= validation_fit_end) & (series.index < validation_end)
    validation_qlike = _qlike(
        series.to_numpy(dtype=float)[validation_mask],
        validation_full_variance[validation_mask],
    )

    method, parameters, variance, errors = _fit_with_fallback(
        production.to_numpy(dtype=float), settings
    )
    full_variance = (
        ewma_variance_recursion(
            series.to_numpy(dtype=float),
            decay=float(settings["ewma_decay"]),
            variance_floor=float(settings["variance_floor"]),
        )
        if method == "ewma"
        else gjr_variance_recursion(
            series.to_numpy(dtype=float),
            **{key: parameters[key] for key in ["omega", "alpha", "gamma", "beta"]},
            variance_floor=float(settings["variance_floor"]),
        )
    )
    sigma = np.sqrt(full_variance)
    residuals = series.to_numpy(dtype=float) / sigma
    diagnostics = {
        "PRODUCTION_METHOD": method,
        "VALIDATION_METHOD": validation_method,
        "VALIDATION_QLIKE": validation_qlike,
        "OMEGA": parameters["omega"],
        "ALPHA": parameters["alpha"],
        "GAMMA": parameters["gamma"],
        "BETA": parameters["beta"],
        "NU": parameters["nu"],
        "PERSISTENCE": parameters["persistence"],
        "PRODUCTION_FALLBACK_ERRORS": errors,
        "VALIDATION_FALLBACK_ERRORS": validation_errors,
        "STANDARDIZED_RESIDUAL_MEAN": float(np.mean(residuals)),
        "STANDARDIZED_RESIDUAL_STD": float(np.std(residuals, ddof=1)),
    }
    return sigma, residuals, diagnostics


def prepare_dynamic_scale_artifacts(config, registry, force=False):
    """Fit/cache stock-level filters and return paths to reusable artifacts."""
    settings = config.get("dynamic_scale", {})
    if not settings.get("enabled", False):
        return None
    paths = _cache_paths(config)
    stock_path = _registry_stock_path(config, registry)
    signature = {
        "source_path": str(stock_path.resolve()),
        "source_sha256": _sha256(stock_path),
        "settings": _effective_config(settings),
    }
    if (
        not force
        and settings.get("reuse_cache", True)
        and all(paths[key].exists() for key in ["sigma", "residuals", "diagnostics", "metadata"])
    ):
        with paths["metadata"].open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("signature") == signature:
            return paths

    stock = pd.read_csv(stock_path, index_col=0, parse_dates=[0]).sort_index()
    tickers = [column for column in stock.columns if column != "STATE"]
    sigma = pd.DataFrame(index=stock.index, columns=tickers, dtype=float)
    residuals = pd.DataFrame(index=stock.index, columns=tickers, dtype=float)
    diagnostics = []
    paths["root"].mkdir(parents=True, exist_ok=True)
    for number, ticker in enumerate(tickers, start=1):
        series = stock[ticker].astype(float)
        ticker_sigma, ticker_residuals, row = _fit_one_stock(series, settings)
        sigma[ticker] = ticker_sigma
        residuals[ticker] = ticker_residuals
        diagnostics.append({"TICKER": ticker, **row})
        print(
            f"[dynamic scale] volatility {number}/{len(tickers)}: "
            f"{ticker} ({row['PRODUCTION_METHOD']})",
            flush=True,
        )

    if not np.isfinite(sigma.to_numpy(dtype=float)).all():
        raise ValueError("Conditional-volatility cache contains non-finite values.")
    if not np.isfinite(residuals.to_numpy(dtype=float)).all():
        raise ValueError("Standardized-residual cache contains non-finite values.")
    sigma.to_csv(paths["sigma"], compression="gzip")
    residuals.to_csv(paths["residuals"], compression="gzip")
    pd.DataFrame(diagnostics).to_csv(paths["diagnostics"], index=False)
    metadata = {
        "signature": signature,
        "n_tickers": len(tickers),
        "n_dates": len(stock),
        "date_start": str(stock.index.min().date()),
        "date_end": str(stock.index.max().date()),
    }
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return paths


def load_dynamic_scale_frames(config, tickers):
    paths = _cache_paths(config)
    sigma = pd.read_csv(paths["sigma"], index_col=0, parse_dates=[0])
    residuals = pd.read_csv(paths["residuals"], index_col=0, parse_dates=[0])
    missing = sorted(set(tickers) - set(sigma.columns))
    if missing:
        raise ValueError(f"Dynamic-scale cache is missing tickers: {missing}")
    return sigma[tickers].sort_index(), residuals[tickers].sort_index()
