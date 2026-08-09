"""Rolling regime-conditional WGAN training and paired one-step simulation."""

from __future__ import annotations

import gc
import json
import os
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_risk import (
    rolling_exceedance_counts,
    save_portfolio_risk_figures,
    score_risk_forecasts,
)
from pipeline_runtime import resolve_project_path
from pipeline_volatility import load_dynamic_scale_frames


def _tensorflow_components():
    """Import TensorFlow only for an actual WGAN run, never for a dry run."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from keras.optimizers import Adam
    from keras.utils import set_random_seed

    from p2_0_utils import break_check, critic, generator, grad_penalty

    return tf, Adam, set_random_seed, break_check, critic, generator, grad_penalty


def _registry_artifact(registry, column, project_root):
    values = [value for value in registry[column].astype(str).unique() if value]
    if len(values) != 1:
        raise ValueError(f"Registry must identify one shared {column} artifact.")
    return resolve_project_path(values[0], project_root)


def _load_portfolio_inputs(config, registry, manifest):
    project_root = Path(config["_project_root"])
    stock_path = _registry_artifact(registry, "production_stock_returns", project_root)
    probability_path = _registry_artifact(
        registry, "transition_probability_file", project_root
    )
    tickers = manifest.sort_values("POSITION")["TICKER"].tolist()
    stock = pd.read_csv(stock_path, index_col=0, parse_dates=[0])
    missing = sorted(set(tickers) - set(stock.columns))
    if missing:
        raise ValueError(f"Production stock returns are missing tickers: {missing}")
    stock = stock[tickers + ["STATE"]].dropna()
    probabilities = pd.read_csv(probability_path, parse_dates=["FORECAST_DATE"])
    probabilities = probabilities.set_index("FORECAST_DATE").sort_index()
    conditional_sigma = None
    training_stock = stock
    if config.get("dynamic_scale", {}).get("enabled", False):
        conditional_sigma, residuals = load_dynamic_scale_frames(config, tickers)
        common = stock.index.intersection(conditional_sigma.index).intersection(
            residuals.index
        )
        if not common.equals(stock.index):
            missing = stock.index.difference(common)
            raise ValueError(
                f"Dynamic-scale cache is missing {len(missing)} stock-return dates."
            )
        conditional_sigma = conditional_sigma.loc[stock.index]
        training_stock = residuals.loc[stock.index].copy()
        training_stock["STATE"] = stock["STATE"].to_numpy(dtype=int)
    return stock, training_stock, conditional_sigma, probabilities, tickers


def _effective_sample_size(weights):
    """Return Kish's effective sample size for non-negative weights."""
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return 0.0
    total = float(weights.sum())
    squared_total = float(np.square(weights).sum())
    if not np.isfinite(total) or not np.isfinite(squared_total) or squared_total <= 0:
        return 0.0
    return total**2 / squared_total


def _duration_text(seconds):
    """Format a duration for compact progress logs."""
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _adaptive_regime_memory(
    stock,
    position,
    state,
    tickers,
    settings,
    *,
    use_full_history=False,
):
    """Build a causal, recency-weighted memory for one HMM regime."""
    history = stock.iloc[:position]
    state_positions = np.flatnonzero(history["STATE"].to_numpy() == state)
    memory_settings = settings["regime_memory"]
    threshold = float(memory_settings["minimum_effective_sample_size"])
    half_life = float(memory_settings["recency_half_life_trading_days"])
    base_window = int(settings["rolling_window_size"])

    if state_positions.size == 0:
        return {
            "data": np.empty((0, len(tickers)), dtype=np.float32),
            "weights": np.empty(0, dtype=float),
            "metadata": {
                "OBSERVATIONS_AVAILABLE": 0,
                "BASE_WINDOW_OBSERVATIONS": 0,
                "EFFECTIVE_SAMPLE_SIZE": 0.0,
                "ESS_THRESHOLD": threshold,
                "RECENCY_HALF_LIFE": half_life,
                "MEMORY_LOOKBACK_ROWS": 0,
                "MEMORY_START_DATE": "",
                "MEMORY_END_DATE": "",
            },
        }

    # Scaling all weights by a common factor leaves both ESS and sampling
    # probabilities unchanged. Anchoring at the newest state observation avoids
    # numerical underflow when a regime has been absent for a long period.
    ages = position - 1 - state_positions
    relative_ages = ages - ages.min()
    all_weights = np.exp2(-relative_ages / half_life)
    reverse_sum = np.cumsum(all_weights[::-1])[::-1]
    reverse_squared_sum = np.cumsum(np.square(all_weights[::-1]))[::-1]
    suffix_ess = np.square(reverse_sum) / reverse_squared_sum

    if use_full_history:
        selected_start = 0
        base_count = int(state_positions.size)
    else:
        base_start_position = max(0, position - base_window)
        base_start = int(np.searchsorted(state_positions, base_start_position))
        base_count = int(state_positions.size - base_start)
        latest_allowed_start = min(base_start, state_positions.size - 1)
        eligible = np.flatnonzero(
            (np.arange(state_positions.size) <= latest_allowed_start)
            & (suffix_ess >= threshold)
        )
        # If the threshold cannot be reached, retain all available state data
        # for diagnostics; the ESS gate will prevent a training update.
        selected_start = int(eligible.max()) if eligible.size else 0

    selected_positions = state_positions[selected_start:]
    selected_weights = all_weights[selected_start:]
    selected_ess = _effective_sample_size(selected_weights)
    selected_frame = history.iloc[selected_positions]
    return {
        "data": selected_frame[tickers].to_numpy(dtype=np.float32),
        "weights": selected_weights,
        "metadata": {
            "OBSERVATIONS_AVAILABLE": int(state_positions.size),
            "BASE_WINDOW_OBSERVATIONS": base_count,
            "EFFECTIVE_SAMPLE_SIZE": selected_ess,
            "ESS_THRESHOLD": threshold,
            "RECENCY_HALF_LIFE": half_life,
            "MEMORY_LOOKBACK_ROWS": int(position - selected_positions[0]),
            "MEMORY_START_DATE": str(selected_frame.index.min()),
            "MEMORY_END_DATE": str(selected_frame.index.max()),
        },
    }


def _build_compiled_train_step(
    tf,
    critic_network,
    generator_network,
    critic_optimizer,
    generator_optimizer,
    latent_dimension,
    critic_steps,
    penalty_weight,
    grad_penalty,
):
    """Compile one full WGAN iteration to avoid per-operation Python dispatch."""

    if hasattr(critic_optimizer, "build"):
        critic_optimizer.build(critic_network.trainable_variables)
    if hasattr(generator_optimizer, "build"):
        generator_optimizer.build(generator_network.trainable_variables)

    @tf.function(reduce_retracing=True)
    def train_iteration(real_data):
        critic_loss = tf.constant(np.nan, dtype=tf.float32)
        latent_shape = tf.stack([
            tf.shape(real_data)[0],
            tf.constant(latent_dimension, dtype=tf.int32),
        ])
        # The original implementation reuses one sampled real batch for all
        # critic steps within a generator iteration.
        for _ in range(critic_steps):
            z = tf.random.normal(latent_shape, dtype=real_data.dtype)
            # Match the source loop: fake samples are generated before the
            # critic tape, so critic updates do not record the large generator
            # graph or retain generator intermediates unnecessarily.
            fake_data = tf.stop_gradient(
                generator_network(z, training=False)
            )
            with tf.GradientTape() as tape:
                # The source implementation calls both networks without
                # training=True.  Be explicit so the critic's Dropout layers
                # remain disabled consistently for scores and gradient penalty.
                real_score = critic_network(real_data, training=False)
                fake_score = critic_network(fake_data, training=False)
                critic_cost = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
                critic_loss = critic_cost + penalty_weight * grad_penalty(
                    real_data, fake_data, critic_network
                )
            gradients = tape.gradient(
                critic_loss, critic_network.trainable_variables
            )
            critic_optimizer.apply_gradients([
                (gradient, variable)
                for gradient, variable in zip(
                    gradients, critic_network.trainable_variables
                )
                if gradient is not None
            ])

        z = tf.random.normal(latent_shape, dtype=real_data.dtype)
        with tf.GradientTape() as tape:
            fake_data = generator_network(z, training=False)
            fake_score = critic_network(fake_data, training=False)
            generator_loss = -tf.reduce_mean(fake_score)
        gradients = tape.gradient(
            generator_loss, generator_network.trainable_variables
        )
        generator_optimizer.apply_gradients([
            (gradient, variable)
            for gradient, variable in zip(
                gradients, generator_network.trainable_variables
            )
            if gradient is not None
        ])
        return critic_loss, generator_loss

    return train_iteration


def _fit_wgan_window(
    state,
    state_data,
    sample_weights,
    networks,
    optimizers,
    settings,
    window_number,
    rng,
    components,
    *,
    loss_history=None,
    compiled_train_step=None,
):
    tf, _, _, break_check, _, _, grad_penalty = components
    critic_network, generator_network = networks[state]
    critic_optimizer, generator_optimizer = optimizers[state]
    batch_size = int(settings["batch_size"])
    latent_dimension = int(settings["latent_dimension"])
    default_iterations = int(settings.get("training_iterations", 2000))
    if window_number == 0:
        iterations = int(
            settings.get("initial_training_iterations", default_iterations)
        )
    else:
        iterations = int(
            settings.get("rolling_training_iterations", default_iterations)
        )
    critic_steps = int(settings["critic_steps"])
    penalty_weight = float(settings["gradient_penalty"])
    ess_threshold = float(
        settings["regime_memory"]["minimum_effective_sample_size"]
    )
    effective_sample_size = _effective_sample_size(sample_weights)
    if len(state_data) == 0 or effective_sample_size < ess_threshold:
        return {
            "STATE": state,
            "WINDOW": window_number,
            "LENGTH": len(state_data),
            "ITERATIONS_COMPLETED": 0,
            "ITERATION_LIMIT": iterations,
            "CRITIC_LOSS": np.nan,
            "GENERATOR_LOSS": np.nan,
            "CRITIC_LOSS_SLOPE": np.nan,
            "UPDATE_APPLIED": False,
            "UPDATE_REASON": "ess_below_threshold",
        }

    if loss_history is None:
        loss_history = deque(maxlen=100)
    critic_loss_value = generator_loss_value = np.nan
    slope = np.nan
    state_data = np.asarray(state_data, dtype=np.float32)
    sampling_probabilities = np.asarray(sample_weights, dtype=float)
    sampling_probabilities /= sampling_probabilities.sum()
    if compiled_train_step is None:
        compiled_train_step = _build_compiled_train_step(
            tf,
            critic_network,
            generator_network,
            critic_optimizer,
            generator_optimizer,
            latent_dimension,
            critic_steps,
            penalty_weight,
            grad_penalty,
        )

    for iteration in range(iterations):
        indexes = rng.choice(
            len(state_data),
            size=batch_size,
            # Weighted bootstrap sampling matches the source implementation's
            # with-replacement batches while respecting adaptive memory weights.
            replace=True,
            p=sampling_probabilities,
        )
        real_data = tf.convert_to_tensor(state_data[indexes])
        critic_loss, generator_loss = compiled_train_step(real_data)
        critic_loss_value = float(critic_loss.numpy())
        generator_loss_value = float(generator_loss.numpy())
        loss_history.append(critic_loss_value)
        recent = np.asarray(loss_history, dtype=float)
        mean_loss = float(recent.mean())
        standard_deviation = float(recent.std())
        if recent.size > 1:
            slope = abs(float(np.polyfit(np.arange(recent.size), recent, 1)[0]))
        if break_check(
            window_number,
            iteration,
            mean_loss,
            standard_deviation,
            slope,
        ):
            break

    return {
        "STATE": state,
        "WINDOW": window_number,
        "LENGTH": len(state_data),
        "ITERATIONS_COMPLETED": iteration + 1,
        "ITERATION_LIMIT": iterations,
        "CRITIC_LOSS": critic_loss_value,
        "GENERATOR_LOSS": generator_loss_value,
        "CRITIC_LOSS_SLOPE": slope,
        "UPDATE_APPLIED": True,
        "UPDATE_REASON": "ess_at_or_above_threshold",
    }


def _sample_states(probabilities, uniforms):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Transition probability row is invalid.")
    cumulative = np.cumsum(probabilities / total)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, uniforms, side="right")


def _simulate_portfolio_returns(
    generators,
    sampled_states,
    latent_draws,
    weights,
    conditional_sigma=None,
):
    simulations = np.empty(len(sampled_states), dtype=float)
    representative = None
    if conditional_sigma is not None:
        conditional_sigma = np.asarray(conditional_sigma, dtype=float)
        if conditional_sigma.shape != weights.shape:
            raise ValueError("Conditional-volatility and portfolio-weight shapes differ.")
        if not np.isfinite(conditional_sigma).all() or (conditional_sigma <= 0).any():
            raise ValueError("Conditional-volatility vector must be finite and positive.")
    for state in np.unique(sampled_states):
        indexes = np.flatnonzero(sampled_states == state)
        generated = generators[int(state)](latent_draws[indexes], training=False).numpy()
        if conditional_sigma is not None:
            generated = generated * conditional_sigma
        simulations[indexes] = generated @ weights
        if 0 in indexes:
            representative = generated[np.flatnonzero(indexes == 0)[0]]
    return simulations, representative


def _risk_row(date, model_id, realized_return, simulations, levels):
    row = {
        "DATE": date,
        "MODEL_ID": model_id,
        "REALIZED_RETURN": float(realized_return),
        "SIMULATION_MEAN": float(np.mean(simulations)),
        "SIMULATION_STD": float(np.std(simulations, ddof=1)),
    }
    for confidence in levels:
        tail_probability = 1.0 - float(confidence)
        suffix = f"{100.0 * float(confidence):g}".replace(".", "")
        var = float(np.quantile(simulations, tail_probability))
        tail = simulations[simulations <= var]
        row[f"VAR_{suffix}"] = var
        row[f"ES_{suffix}"] = float(tail.mean()) if tail.size else var
    return row


def _save_checkpoints(generators, directory, label):
    directory = Path(directory) / label
    directory.mkdir(parents=True, exist_ok=True)
    for state, network in generators.items():
        network.save_weights(directory / f"generator_state_{state}.weights.h5")


def _save_loss_plot(loss_report, path):
    if loss_report.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for state, frame in loss_report.groupby("STATE"):
        axis.plot(frame["DATE"], frame["CRITIC_LOSS"], linewidth=0.6, label=f"State {state}")
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_title("Final critic loss by rolling WGAN update")
    axis.set_ylabel("Critic loss")
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_ess_plot(loss_report, path):
    if loss_report.empty or "EFFECTIVE_SAMPLE_SIZE" not in loss_report:
        return
    states = sorted(loss_report["STATE"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for axis, state in zip(axes.flat, states):
        frame = loss_report[loss_report["STATE"].eq(state)].sort_values("DATE")
        threshold = float(frame["ESS_THRESHOLD"].iloc[0])
        applied = frame["UPDATE_APPLIED"].astype(bool)
        axis.plot(
            frame["DATE"],
            frame["EFFECTIVE_SAMPLE_SIZE"],
            color="#1f5a85",
            linewidth=0.75,
        )
        axis.scatter(
            frame.loc[~applied, "DATE"],
            frame.loc[~applied, "EFFECTIVE_SAMPLE_SIZE"],
            color="#b44620",
            marker="x",
            s=10,
            linewidths=0.7,
            label="Update frozen",
        )
        axis.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=0.8,
            label=f"ESS gate = {threshold:g}",
        )
        axis.set_title(f"State {state}")
        axis.grid(alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_ylabel("Weighted ESS")
    for axis in axes[-1, :]:
        axis.set_xlabel("Update date")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Adaptive regime-memory ESS and frozen WGAN updates", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_portfolio_pipeline(config, registry, manifest):
    """Train state WGANs shared by the fixed and dynamic transition models."""
    sample_id = int(manifest["SAMPLE_ID"].iloc[0])
    output_root = resolve_project_path(
        config["outputs"]["root_directory"], config["_project_root"]
    )
    output_dir = output_root / f"portfolio_{sample_id:02d}"
    success_path = output_dir / "SUCCESS.json"
    if config["execution"].get("resume_completed_work", True) and success_path.exists():
        return {
            "sample_id": sample_id,
            "output_dir": output_dir,
            "status": "already_complete",
            "scores_path": output_dir / "risk_scores.csv",
        }
    if output_dir.exists() and not config["execution"].get("overwrite_existing_outputs", False):
        existing = [path for path in output_dir.iterdir() if path.name != "FAILURE.json"]
        if existing:
            raise FileExistsError(
                f"Portfolio {sample_id} has incomplete outputs. Enable overwrite or remove them: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    components = _tensorflow_components()
    tf, Adam, set_random_seed, _, critic, generator, _ = components
    wgan = config["wgan"]
    seed = int(wgan["random_seed"]) + sample_id
    set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    rng = np.random.default_rng(seed)
    simulation_rng = np.random.default_rng(int(config["simulation"]["random_seed"]) + sample_id)
    progress_interval = int(config["execution"].get("progress_interval_dates", 10))
    if progress_interval < 1:
        raise ValueError("execution.progress_interval_dates must be positive.")
    cleanup_interval = int(
        config["execution"].get("memory_cleanup_interval_dates", 50)
    )
    portfolio_started = time.monotonic()

    stock, training_stock, conditional_sigma, probabilities, tickers = (
        _load_portfolio_inputs(config, registry, manifest)
    )
    n_states = int(config["phase_1"]["n_regimes"])
    n_stocks = len(tickers)
    weights = manifest.sort_values("POSITION")["WEIGHT"].to_numpy(dtype=float)
    initial_end = pd.Timestamp(wgan["initial_train_end_exclusive"])
    forecast_start = pd.Timestamp(config["data"]["forecast_start"])
    forecast_end = config["data"]["forecast_end"]
    if forecast_end != "auto_common_end":
        stock = stock[stock.index <= pd.Timestamp(forecast_end)]
        probabilities = probabilities[probabilities.index <= pd.Timestamp(forecast_end)]
    initial_count = int((stock.index < initial_end).sum())
    if initial_count == 0:
        raise ValueError("No observations are available before the initial WGAN cutoff.")
    forecast_dates = stock.index[stock.index >= forecast_start]
    missing_probabilities = forecast_dates.difference(probabilities.index)
    if len(missing_probabilities):
        raise ValueError(
            f"Transition probabilities are missing {len(missing_probabilities)} forecast dates; first missing date is {missing_probabilities[0].date()}."
        )

    networks, optimizers = {}, {}
    generator_output_activation = (
        "linear"
        if config.get("dynamic_scale", {}).get("enabled", False)
        else "tanh"
    )
    for state in range(n_states):
        networks[state] = (
            critic(n_stocks),
            generator(
                int(wgan["latent_dimension"]),
                n_stocks,
                output_activation=generator_output_activation,
            ),
        )
        optimizers[state] = (
            Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9),
            Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9),
        )
    generators = {state: pair[1] for state, pair in networks.items()}
    compiled_train_steps = {}
    for state in range(n_states):
        critic_network, generator_network = networks[state]
        critic_optimizer, generator_optimizer = optimizers[state]
        compiled_train_steps[state] = _build_compiled_train_step(
            tf,
            critic_network,
            generator_network,
            critic_optimizer,
            generator_optimizer,
            int(wgan["latent_dimension"]),
            int(wgan["critic_steps"]),
            float(wgan["gradient_penalty"]),
            components[-1],
        )

    loss_rows, risk_rows, state_count_rows = [], [], []
    loss_histories = {
        state: deque(maxlen=100) for state in range(n_states)
    }
    dynamic_id = str(
        registry.loc[registry["role"].eq("dynamic"), "model_id"].iloc[0]
    )
    transition_models = config["simulation"].get(
        "transition_models", ["fixed", "dynamic"]
    )
    model_specs = []
    if "fixed" in transition_models:
        model_specs.append(("fixed_hmm_transition", "FIXED"))
    if "dynamic" in transition_models:
        model_specs.append((dynamic_id, "DYNAMIC"))
    if not model_specs:
        raise ValueError("At least one transition model must be enabled.")
    representative = {model_id: [] for model_id, _ in model_specs}
    initialized = stock.iloc[:initial_count]
    print(
        f"[portfolio {sample_id:02d}] Initializing {n_states} regime WGANs "
        f"with {n_stocks} stocks; cutoff={initial_end.date()}",
        flush=True,
    )
    for state in range(n_states):
        memory = _adaptive_regime_memory(
            training_stock,
            initial_count,
            state,
            tickers,
            wgan,
            use_full_history=True,
        )
        result = _fit_wgan_window(
            state,
            memory["data"],
            memory["weights"],
            networks,
            optimizers,
            wgan,
            0,
            rng,
            components,
            loss_history=loss_histories[state],
            compiled_train_step=compiled_train_steps[state],
        )
        result.update(memory["metadata"])
        result["DATE"] = initialized.index.max()
        result["UPDATE_TYPE"] = "initial"
        loss_rows.append(result)
        print(
            f"[portfolio {sample_id:02d}] Initial state {state}: "
            f"observations={result['LENGTH']}, "
            f"ESS={result['EFFECTIVE_SAMPLE_SIZE']:.2f}, "
            f"iterations={result['ITERATIONS_COMPLETED']}, "
            f"applied={result['UPDATE_APPLIED']}",
            flush=True,
        )
        if not result["UPDATE_APPLIED"]:
            raise ValueError(
                f"State {state} has initial weighted ESS "
                f"{result['EFFECTIVE_SAMPLE_SIZE']:.2f}, below the required "
                f"threshold {result['ESS_THRESHOLD']:.0f}; a valid initial "
                "state WGAN cannot be trained."
            )

    warmup_checkpoint_saved = False
    levels = config["evaluation"]["original_risk_levels"]
    path_count = int(config["simulation"]["paths_per_forecast_date"])
    total_dates = len(stock) - initial_count
    forecast_date_count = int((stock.index >= forecast_start).sum())
    print(
        f"[portfolio {sample_id:02d}] Rolling run started: dates={total_dates}, "
        f"forecast_dates={forecast_date_count}, progress_interval={progress_interval}",
        flush=True,
    )
    rolling_started = time.monotonic()
    for progress_number, position in enumerate(
        range(initial_count, len(stock)), start=1
    ):
        date = stock.index[position]
        rolling_window = int(wgan["rolling_window_size"])
        window = stock.iloc[max(0, position - rolling_window):position]
        # The first post-cutoff forecast uses the initial fit. Subsequent updates
        # contain only observations available before the date being forecast.
        if position > initial_count:
            for state in range(n_states):
                memory = _adaptive_regime_memory(
                    training_stock,
                    position,
                    state,
                    tickers,
                    wgan,
                )
                result = _fit_wgan_window(
                    state,
                    memory["data"],
                    memory["weights"],
                    networks,
                    optimizers,
                    wgan,
                    position - initial_count,
                    rng,
                    components,
                    loss_history=loss_histories[state],
                    compiled_train_step=compiled_train_steps[state],
                )
                result.update(memory["metadata"])
                result["DATE"] = date
                result["UPDATE_TYPE"] = "rolling"
                loss_rows.append(result)

        if (
            progress_number == 1
            or progress_number % progress_interval == 0
            or progress_number == total_dates
        ):
            elapsed = time.monotonic() - rolling_started
            remaining = (elapsed / progress_number) * (total_dates - progress_number)
            applied_updates = sum(bool(row["UPDATE_APPLIED"]) for row in loss_rows)
            rolling_results = [
                row for row in loss_rows if row["UPDATE_TYPE"] == "rolling"
            ]
            rolling_iterations = [
                row["ITERATIONS_COMPLETED"]
                for row in rolling_results
                if row["UPDATE_APPLIED"]
            ]
            average_iterations = (
                float(np.mean(rolling_iterations)) if rolling_iterations else 0.0
            )
            print(
                f"[portfolio {sample_id:02d}] Progress {progress_number}/{total_dates} "
                f"({100.0 * progress_number / total_dates:.2f}%); "
                f"date={date.date()}; "
                f"stage={'forecast' if date >= forecast_start else 'warmup'}; "
                f"elapsed={_duration_text(elapsed)}; "
                f"ETA={_duration_text(remaining)}; "
                f"avg_rolling_iterations={average_iterations:.2f}; "
                f"updates_applied={applied_updates}; "
                f"updates_skipped={len(loss_rows) - applied_updates}",
                flush=True,
            )

        if progress_number % cleanup_interval == 0:
            # Compiled train functions and model variables remain live; this
            # only collects unreachable Python/NumPy/TensorFlow wrapper objects.
            gc.collect()

        if date < forecast_start:
            continue
        if not warmup_checkpoint_saved and config["execution"].get("save_wgan_checkpoints", True):
            _save_checkpoints(generators, output_dir / "wgan_checkpoints", "through_2016")
            warmup_checkpoint_saved = True

        probability_row = probabilities.loc[date]
        uniforms = simulation_rng.random(path_count)
        latent = simulation_rng.standard_normal(
            (path_count, int(wgan["latent_dimension"]))
        ).astype(np.float32)
        realized_return = float(stock.loc[date, tickers].to_numpy(dtype=float) @ weights)
        sigma_vector = (
            conditional_sigma.loc[date, tickers].to_numpy(dtype=float)
            if conditional_sigma is not None
            else None
        )
        for model_id, prefix in model_specs:
            probability_vector = np.array(
                [probability_row[f"{prefix}_P_TO_{state}"] for state in range(n_states)],
                dtype=float,
            )
            sampled_states = _sample_states(probability_vector, uniforms)
            simulations, representative_return = _simulate_portfolio_returns(
                generators,
                sampled_states,
                latent,
                weights,
                conditional_sigma=sigma_vector,
            )
            risk_rows.append(
                _risk_row(date, model_id, realized_return, simulations, levels)
            )
            representative[model_id].append(
                {"DATE": date, **dict(zip(tickers, representative_return))}
            )
            counts = np.bincount(sampled_states, minlength=n_states)
            state_count_rows.extend([
                {
                    "DATE": date,
                    "MODEL_ID": model_id,
                    "STATE": state,
                    "PROBABILITY": probability_vector[state],
                    "SIMULATED_COUNT": int(counts[state]),
                }
                for state in range(n_states)
            ])

        if config["simulation"].get("include_normal_benchmark", True):
            normal_window = window[tickers].to_numpy(dtype=float) @ weights
            normal_simulations = simulation_rng.normal(
                float(np.mean(normal_window)),
                float(np.std(normal_window, ddof=1)),
                path_count,
            )
            risk_rows.append(
                _risk_row(date, "normal", realized_return, normal_simulations, levels)
            )
        elif config["simulation"].get("replay_fixed_baseline_rng_sequence", False):
            # The preserved fixed run simulated a normal benchmark after each
            # date. Discarding the same draw keeps future uniforms and latent
            # vectors paired across the old and new experiments.
            simulation_rng.normal(0.0, 1.0, path_count)

    loss_report = pd.DataFrame(loss_rows)
    forecasts = pd.DataFrame(risk_rows)
    state_counts = pd.DataFrame(state_count_rows)
    print(
        f"[portfolio {sample_id:02d}] Training and simulation complete after "
        f"{_duration_text(time.monotonic() - portfolio_started)}; saving outputs.",
        flush=True,
    )
    scores = score_risk_forecasts(forecasts, config, sample_id)
    rolling = rolling_exceedance_counts(forecasts)
    loss_report.to_csv(output_dir / "wgan_loss_report.csv", index=False)
    forecasts.to_csv(output_dir / "daily_risk_forecasts.csv", index=False)
    scores.to_csv(output_dir / "risk_scores.csv", index=False)
    rolling.to_csv(output_dir / "rolling_250_day_exceedances.csv", index=False)
    state_counts.to_csv(output_dir / "simulated_state_counts.csv", index=False)
    stock[tickers + ["STATE"]].to_csv(output_dir / "portfolio_real_returns.csv")
    for model_id, rows in representative.items():
        pd.DataFrame(rows).set_index("DATE").to_csv(
            output_dir / f"representative_simulated_returns_{model_id}.csv"
        )
    if conditional_sigma is not None:
        forecast_index = pd.DatetimeIndex(
            pd.to_datetime(forecasts["DATE"]).unique()
        ).sort_values()
        forecast_sigma = conditional_sigma.loc[forecast_index, tickers]
        sigma_summary = pd.DataFrame({
            "DATE": forecast_sigma.index,
            "MEAN_STOCK_SIGMA": forecast_sigma.mean(axis=1).to_numpy(),
            "MEDIAN_STOCK_SIGMA": forecast_sigma.median(axis=1).to_numpy(),
            "MIN_STOCK_SIGMA": forecast_sigma.min(axis=1).to_numpy(),
            "MAX_STOCK_SIGMA": forecast_sigma.max(axis=1).to_numpy(),
            "WEIGHTED_MEAN_STOCK_SIGMA": (
                forecast_sigma.to_numpy(dtype=float) @ weights
            ),
        })
        sigma_summary.to_csv(
            output_dir / "dynamic_scale_daily_summary.csv", index=False
        )
    _save_loss_plot(loss_report, output_dir / "wgan_critic_loss.png")
    _save_ess_plot(loss_report, output_dir / "wgan_ess_diagnostics.png")
    if config["outputs"].get("save_original_paper_figures", True):
        save_portfolio_risk_figures(
            forecasts, rolling, output_dir / "original_paper_risk_diagnostics"
        )
        from p3_1_stat_prop import p3_1_stat_prop
        from p3_2_stat_prop_mv import p3_2_stat_prop_mv

        diagnostics_dir = output_dir / "original_paper_stylized_facts"
        stock_path = output_dir / "portfolio_real_returns.csv"
        for model_id in representative:
            simulation_path = output_dir / f"representative_simulated_returns_{model_id}.csv"
            arguments = {
                "n_states": n_states,
                "n_stocks": n_stocks,
                "stock_rets_path": stock_path,
                "sim_rets_path": simulation_path,
                "output_dir": diagnostics_dir,
                "train_date": str(forecast_start.date()),
                "output_suffix": model_id,
            }
            p3_1_stat_prop(**arguments)
            p3_2_stat_prop_mv(**arguments)
    if config["execution"].get("save_wgan_checkpoints", True):
        _save_checkpoints(generators, output_dir / "wgan_checkpoints", "final")

    with success_path.open("w", encoding="utf-8") as handle:
        applied_rolling = loss_report[
            loss_report["UPDATE_TYPE"].eq("rolling")
            & loss_report["UPDATE_APPLIED"].astype(bool)
        ]
        json.dump({
            "sample_id": sample_id,
            "dynamic_model_id": dynamic_id,
            "stock_count": n_stocks,
            "wgan_training_mode": wgan["training_mode"],
            "dynamic_scale_enabled": bool(
                config.get("dynamic_scale", {}).get("enabled", False)
            ),
            "generator_output_activation": generator_output_activation,
            "transition_models_run": [model_id for model_id, _ in model_specs],
            "wgan_ess_update_threshold": wgan["regime_memory"][
                "minimum_effective_sample_size"
            ],
            "wgan_recency_half_life_trading_days": wgan["regime_memory"][
                "recency_half_life_trading_days"
            ],
            "wgan_initial_iteration_limit": int(
                wgan.get("initial_training_iterations", wgan["training_iterations"])
            ),
            "wgan_rolling_iteration_limit": int(
                wgan.get("rolling_training_iterations", wgan["training_iterations"])
            ),
            "wgan_average_applied_rolling_iterations": float(
                applied_rolling["ITERATIONS_COMPLETED"].mean()
            ) if not applied_rolling.empty else 0.0,
            "wgan_median_applied_rolling_iterations": float(
                applied_rolling["ITERATIONS_COMPLETED"].median()
            ) if not applied_rolling.empty else 0.0,
            "forecast_start": str(forecasts["DATE"].min()),
            "forecast_end": str(forecasts["DATE"].max()),
            "forecast_observations_per_model": int(
                len(forecasts) / forecasts["MODEL_ID"].nunique()
            ),
            "simulation_paths_per_date": path_count,
            "elapsed_seconds": float(time.monotonic() - portfolio_started),
        }, handle, indent=2)
    print(
        f"[portfolio {sample_id:02d}] SUCCESS written to {success_path}",
        flush=True,
    )
    return {
        "sample_id": sample_id,
        "output_dir": output_dir,
        "status": "complete",
        "scores_path": output_dir / "risk_scores.csv",
    }
