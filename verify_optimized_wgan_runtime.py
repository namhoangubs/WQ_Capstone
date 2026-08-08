"""Exercise the compiled WGAN training path before a long Athena run."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import set_random_seed

from p2_0_utils import critic, generator, grad_penalty
from pipeline_wgan import _build_compiled_train_step


def _variable_device(variable):
    """Return the device string for both TensorFlow and Keras 3 variables."""
    value = getattr(variable, "value", variable)
    return str(getattr(value, "device", ""))


def main():
    with Path("pipeline_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    settings = config["wgan"]
    n_stocks = int(config["portfolio_sampling"]["stocks_per_sample"])
    latent_dimension = int(settings["latent_dimension"])
    batch_size = int(settings["batch_size"])

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError(f"Expected exactly one allocated GPU, found {len(gpus)}")

    set_random_seed(int(settings["random_seed"]))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    critic_network = critic(n_stocks)
    generator_network = generator(latent_dimension, n_stocks)
    critic_parameters = int(critic_network.count_params())
    generator_parameters = int(generator_network.count_params())
    expected_critic_parameters = 225_349
    expected_generator_parameters = 11_386_674
    if critic_parameters != expected_critic_parameters:
        raise RuntimeError(
            "Unexpected critic parameter count: "
            f"{critic_parameters:,} != {expected_critic_parameters:,}. "
            "The parameter-free Maxout implementation may have changed."
        )
    if generator_parameters != expected_generator_parameters:
        raise RuntimeError(
            "Unexpected generator parameter count: "
            f"{generator_parameters:,} != {expected_generator_parameters:,}."
        )

    critic_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    generator_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    train_iteration = _build_compiled_train_step(
        tf,
        critic_network,
        generator_network,
        critic_optimizer,
        generator_optimizer,
        latent_dimension,
        int(settings["critic_steps"]),
        float(settings["gradient_penalty"]),
        grad_penalty,
    )

    rng = np.random.default_rng(int(settings["random_seed"]))
    real_data = tf.convert_to_tensor(
        rng.normal(0.0, 0.01, size=(batch_size, n_stocks)).astype(np.float32)
    )
    compile_started = time.monotonic()
    losses = []
    critic_loss, generator_loss = train_iteration(real_data)
    first_values = (float(critic_loss.numpy()), float(generator_loss.numpy()))
    first_iteration_seconds = time.monotonic() - compile_started
    if not np.isfinite(first_values).all():
        raise RuntimeError(f"Compiled WGAN returned non-finite losses: {first_values}")
    losses.append(first_values)

    steady_measurements = 3
    steady_started = time.monotonic()
    for _ in range(steady_measurements):
        critic_loss, generator_loss = train_iteration(real_data)
        values = (float(critic_loss.numpy()), float(generator_loss.numpy()))
        if not np.isfinite(values).all():
            raise RuntimeError(f"Compiled WGAN returned non-finite losses: {values}")
        losses.append(values)
    steady_iteration_seconds = (
        time.monotonic() - steady_started
    ) / steady_measurements

    registry_path = Path(
        config["phase_1"]["later_phase_artifacts"]["selected_model_registry"]
    )
    registry = pd.read_csv(registry_path)
    stock_paths = registry["production_stock_returns"].dropna().astype(str).unique()
    if len(stock_paths) != 1:
        raise RuntimeError("The selected-model registry must identify one stock-return file.")
    stock_dates = pd.read_csv(stock_paths[0], index_col=0, parse_dates=[0]).index
    initial_end = pd.Timestamp(settings["initial_train_end_exclusive"])
    initial_count = int((stock_dates < initial_end).sum())
    rolling_dates = max(int(len(stock_dates) - initial_count), 0)
    n_states = int(config["phase_1"]["n_regimes"])
    maximum_training_iterations = (
        n_states * int(settings["initial_training_iterations"])
        + n_states
        * max(rolling_dates - 1, 0)
        * int(settings["rolling_training_iterations"])
    )
    compile_overhead_seconds = max(
        first_iteration_seconds - steady_iteration_seconds, 0.0
    ) * n_states
    estimated_max_training_hours = (
        maximum_training_iterations * steady_iteration_seconds
        + compile_overhead_seconds
    ) / 3600.0

    devices = sorted({_variable_device(v) for v in critic_network.trainable_variables})
    if not any("GPU:" in device.upper() for device in devices):
        raise RuntimeError(f"WGAN variables were not placed on the GPU: {devices}")

    print("Optimized WGAN runtime verification passed.")
    print("TensorFlow version:", tf.__version__)
    print("TensorFlow GPU:", gpus[0])
    print("Variable devices:", devices)
    print("Critic parameters:", f"{critic_parameters:,}")
    print("Generator parameters:", f"{generator_parameters:,}")
    print("First iteration including graph compile seconds:", f"{first_iteration_seconds:.3f}")
    print("Steady compiled iteration seconds:", f"{steady_iteration_seconds:.3f}")
    print("Maximum configured WGAN iterations:", maximum_training_iterations)
    print(
        "Estimated maximum training-only hours:",
        f"{estimated_max_training_hours:.2f}",
        "(simulation, I/O, and plotting excluded)",
    )
    print("Losses:", losses)


if __name__ == "__main__":
    main()
