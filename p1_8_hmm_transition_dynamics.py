"""Pair-specific abrupt-versus-gradual transition diagnostics for production HMMs."""

import argparse
import os
from pathlib import Path
import warnings

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import wilcoxon

from p1_4_dynamic_transition import TRAIN_END, VALIDATION_END
from project_config import BASE_DIR


DEFAULT_STATES = (2, 4)
DEFAULT_WINDOW_SIZE = 256
DEFAULT_EVENT_RADIUS = 10
DEFAULT_ROLLING_WINDOWS = (5, 20)
DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_BLOCK_LENGTH = 20
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_EVENTS = 5
OUTPUT_DIR = BASE_DIR / 'phase_1' / 'hmm_transition_dynamics'


def production_features(n_states, exog_rets):
    if n_states == 2:
        return ['SPXT', 'LT11TRUU']
    return list(exog_rets.columns)


def fit_production_hmm(n_states, observations, seed):
    model = GaussianHMM(
        n_states,
        'full',
        random_state=int(seed),
        n_iter=10000,
        tol=1e-5,
        implementation='scaling',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(observations)
    return model


def reproduce_production_states(model, observations, n_train, window_size=DEFAULT_WINDOW_SIZE):
    """Reproduce p1_2_hmm's full-train and rolling-window Viterbi labels."""
    values = np.asarray(observations, dtype=float)
    states = list(model.predict(values[:n_train]))
    for end in range(n_train, len(values)):
        start = max(0, end - window_size + 1)
        states.append(int(model.predict(values[start:end + 1])[-1]))
    return np.asarray(states, dtype=int)


def windowed_filtered_probabilities(model, observations, window_size=DEFAULT_WINDOW_SIZE):
    """One-sided state probabilities using the same trailing window as production decoding."""
    values = np.asarray(observations, dtype=float)
    probabilities = np.zeros((len(values), model.n_components), dtype=float)
    for end in range(len(values)):
        start = max(0, end - window_size + 1)
        # At the final row, forward-backward smoothing equals the one-sided filter.
        probabilities[end] = model.predict_proba(values[start:end + 1])[-1]

    return probabilities


def posterior_frame(index, hard_states, probabilities):
    frame = pd.DataFrame(index=index)
    frame.index.name = 'DATE'
    frame['HARD_STATE'] = np.asarray(hard_states, dtype=int)
    for state in range(probabilities.shape[1]):
        frame[f'POSTERIOR_STATE_{state}'] = probabilities[:, state]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-300, None))).sum(axis=1)
    frame['MAX_POSTERIOR'] = probabilities.max(axis=1)
    frame['POSTERIOR_ENTROPY'] = entropy
    frame['NORMALIZED_POSTERIOR_ENTROPY'] = entropy / np.log(probabilities.shape[1])
    return frame


def rolling_moment_frame(observations, n_train, windows=DEFAULT_ROLLING_WINDOWS):
    """Trailing observable moments on training-standardized returns."""
    training = observations.iloc[:n_train]
    scale = training.std(ddof=0).replace(0, 1.0)
    standardized = (observations - training.mean()) / scale
    columns = {}
    for window in windows:
        rolling = standardized.rolling(int(window), min_periods=int(window))
        mean = rolling.mean()
        standard_deviation = rolling.std(ddof=0)
        for feature in observations.columns:
            columns[f'ROLLING_MEAN_Z_W{window}_{feature}'] = mean[feature]
            columns[f'ROLLING_STD_RATIO_W{window}_{feature}'] = standard_deviation[feature]
    return pd.DataFrame(columns, index=observations.index)


def build_event_observations(
    n_states,
    states,
    posterior,
    moments,
    event_radius=DEFAULT_EVENT_RADIUS,
):
    values = states.to_numpy(dtype=int)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    records = []
    counts = {}

    for event_id, location in enumerate(changes, start=1):
        source = int(values[location - 1])
        destination = int(values[location])
        key = (source, destination)
        counts.setdefault(key, {'TOTAL_EVENTS': 0, 'ELIGIBLE_EVENTS': 0})
        counts[key]['TOTAL_EVENTS'] += 1
        eligible = location - event_radius >= 0 and location + event_radius < len(states)
        if not eligible:
            continue
        counts[key]['ELIGIBLE_EVENTS'] += 1

        for relative_day in range(-event_radius, event_radius + 1):
            row_location = location + relative_day
            row = {
                'N_STATES': n_states,
                'EVENT_ID': event_id,
                'TRANSITION_DATE': states.index[location],
                'FROM_STATE': source,
                'TO_STATE': destination,
                'RELATIVE_DAY': relative_day,
                'DESTINATION_POSTERIOR': posterior.iloc[row_location][f'POSTERIOR_STATE_{destination}'],
                'SOURCE_POSTERIOR': posterior.iloc[row_location][f'POSTERIOR_STATE_{source}'],
                'MAX_POSTERIOR': posterior.iloc[row_location]['MAX_POSTERIOR'],
                'NORMALIZED_POSTERIOR_ENTROPY': posterior.iloc[row_location][
                    'NORMALIZED_POSTERIOR_ENTROPY'
                ],
            }
            row.update(moments.iloc[row_location].to_dict())
            records.append(row)

    count_rows = []
    for (source, destination), values_for_pair in sorted(counts.items()):
        count_rows.append({
            'N_STATES': n_states,
            'FROM_STATE': source,
            'TO_STATE': destination,
            **values_for_pair,
        })
    return pd.DataFrame(records), pd.DataFrame(count_rows)


def _bootstrap_interval(values, statistic=np.median, n_bootstrap=1000, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or n_bootstrap <= 0:
        estimate = float(statistic(values))
        return estimate, estimate
    rng = np.random.default_rng() if rng is None else rng
    draws = rng.choice(values, size=(int(n_bootstrap), len(values)), replace=True)
    statistics = np.apply_along_axis(statistic, 1, draws)
    return tuple(np.quantile(statistics, [0.025, 0.975]))


def posterior_event_summary(events, n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES, random_seed=42):
    rows = []
    rng = np.random.default_rng(random_seed)
    group_columns = ['N_STATES', 'FROM_STATE', 'TO_STATE', 'RELATIVE_DAY']
    for keys, group in events.groupby(group_columns):
        values = group['DESTINATION_POSTERIOR'].to_numpy(dtype=float)
        lower, upper = _bootstrap_interval(values, n_bootstrap=n_bootstrap, rng=rng)
        rows.append({
            **dict(zip(group_columns, keys)),
            'EVENT_COUNT': len(values),
            'MEDIAN_DESTINATION_POSTERIOR': float(np.median(values)),
            'BOOTSTRAP_CI_LOWER': lower,
            'BOOTSTRAP_CI_UPPER': upper,
        })
    return pd.DataFrame(rows)


def rolling_event_summary(events, rolling_windows=DEFAULT_ROLLING_WINDOWS):
    rows = []
    id_columns = ['N_STATES', 'FROM_STATE', 'TO_STATE', 'RELATIVE_DAY']
    for window in rolling_windows:
        prefixes = [f'ROLLING_MEAN_Z_W{window}_', f'ROLLING_STD_RATIO_W{window}_']
        for prefix in prefixes:
            statistic = 'rolling_mean_z' if 'MEAN' in prefix else 'rolling_std_ratio'
            for column in [name for name in events.columns if name.startswith(prefix)]:
                feature = column[len(prefix):]
                for keys, group in events.groupby(id_columns):
                    values = group[column].dropna().to_numpy(dtype=float)
                    if len(values) == 0:
                        continue
                    rows.append({
                        **dict(zip(id_columns, keys)),
                        'ROLLING_WINDOW': int(window),
                        'FEATURE': feature,
                        'STATISTIC': statistic,
                        'EVENT_COUNT': len(values),
                        'MEDIAN': float(np.median(values)),
                        'PERCENTILE_25': float(np.quantile(values, 0.25)),
                        'PERCENTILE_75': float(np.quantile(values, 0.75)),
                    })
    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def transition_monotonicity_tests(
    events,
    event_radius=DEFAULT_EVENT_RADIUS,
    min_events=DEFAULT_MIN_EVENTS,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed=42,
):
    rows = []
    rng = np.random.default_rng(random_seed)
    pre_days = np.arange(-event_radius, 0)

    for (n_states, source, destination), group in events.groupby(
        ['N_STATES', 'FROM_STATE', 'TO_STATE']
    ):
        slopes = []
        nondecreasing = []
        boundary_jumps = []
        abrupt = []
        ambiguous = []
        for _, event in group.groupby('EVENT_ID'):
            ordered = event.set_index('RELATIVE_DAY').sort_index()
            if not set(pre_days).issubset(ordered.index) or not {-1, 0}.issubset(ordered.index):
                continue
            path = ordered.loc[pre_days, 'DESTINATION_POSTERIOR'].to_numpy(dtype=float)
            if not np.all(np.isfinite(path)):
                continue
            slopes.append(float(np.polyfit(pre_days, path, 1)[0]))
            nondecreasing.append(bool(np.all(np.diff(path) >= -1e-10)))
            p_before = float(ordered.loc[-1, 'DESTINATION_POSTERIOR'])
            p_at = float(ordered.loc[0, 'DESTINATION_POSTERIOR'])
            boundary_jumps.append(p_at - p_before)
            abrupt.append(p_before < 0.2 and p_at > 0.8)
            ambiguous.append(
                float(ordered.loc[-1, 'MAX_POSTERIOR']) < 0.8
                or float(ordered.loc[0, 'MAX_POSTERIOR']) < 0.8
            )

        slopes = np.asarray(slopes, dtype=float)
        reliable = len(slopes) >= min_events
        if reliable and np.any(slopes != 0):
            p_value = float(wilcoxon(slopes, alternative='greater').pvalue)
        else:
            p_value = np.nan
        lower, upper = _bootstrap_interval(slopes, n_bootstrap=n_bootstrap, rng=rng)
        rows.append({
            'N_STATES': n_states,
            'FROM_STATE': source,
            'TO_STATE': destination,
            'EVENT_COUNT': len(slopes),
            'PRE_TRANSITION_DAYS': f'-{event_radius} to -1',
            'MEDIAN_PRE_TRANSITION_SLOPE_PER_DAY': float(np.median(slopes)) if len(slopes) else np.nan,
            'BOOTSTRAP_SLOPE_CI_LOWER': lower,
            'BOOTSTRAP_SLOPE_CI_UPPER': upper,
            'SHARE_POSITIVE_PRE_TRANSITION_SLOPE': float(np.mean(slopes > 0)) if len(slopes) else np.nan,
            'SHARE_STRICTLY_NONDECREASING_PATHS': float(np.mean(nondecreasing)) if nondecreasing else np.nan,
            'MEDIAN_BOUNDARY_JUMP': float(np.median(boundary_jumps)) if boundary_jumps else np.nan,
            'ABRUPT_20_TO_80_SHARE': float(np.mean(abrupt)) if abrupt else np.nan,
            'AMBIGUOUS_BOUNDARY_SHARE': float(np.mean(ambiguous)) if ambiguous else np.nan,
            'ONE_SIDED_WILCOXON_P_VALUE': p_value,
            'RELIABLE_FOR_5_PERCENT_DECISION': reliable,
            'NULL_HYPOTHESIS': 'median pre-transition destination-probability slope is not positive',
        })

    result = pd.DataFrame(rows)
    result['FDR_ADJUSTED_P_VALUE'] = _benjamini_hochberg(result['ONE_SIDED_WILCOXON_P_VALUE'])
    result['GRADUAL_BUILDUP_SUPPORTED_AT_5_PERCENT'] = (
        result['RELIABLE_FOR_5_PERCENT_DECISION']
        & (result['FDR_ADJUSTED_P_VALUE'] < 0.05)
        & (result['BOOTSTRAP_SLOPE_CI_LOWER'] > 0)
    )
    return result


def observable_shift_tests(
    events,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
    event_radius=DEFAULT_EVENT_RADIUS,
    min_events=DEFAULT_MIN_EVENTS,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed=42,
):
    """Paired event tests for changes in trailing means and volatilities near a switch."""
    rows = []
    rng = np.random.default_rng(random_seed)
    early_days = list(range(-event_radius, -event_radius // 2))
    boundary_days = [-2, -1, 0]

    for window in rolling_windows:
        for statistic, prefix in [
            ('rolling_mean_z', f'ROLLING_MEAN_Z_W{window}_'),
            ('rolling_std_ratio', f'ROLLING_STD_RATIO_W{window}_'),
        ]:
            for column in [name for name in events.columns if name.startswith(prefix)]:
                feature = column[len(prefix):]
                for (n_states, source, destination), group in events.groupby(
                    ['N_STATES', 'FROM_STATE', 'TO_STATE']
                ):
                    differences = []
                    for _, event in group.groupby('EVENT_ID'):
                        ordered = event.set_index('RELATIVE_DAY')
                        if not set(early_days + boundary_days).issubset(ordered.index):
                            continue
                        early = ordered.loc[early_days, column].mean()
                        boundary = ordered.loc[boundary_days, column].mean()
                        if np.isfinite(early) and np.isfinite(boundary):
                            differences.append(float(boundary - early))
                    differences = np.asarray(differences, dtype=float)
                    reliable = len(differences) >= min_events
                    if reliable and np.any(differences != 0):
                        p_value = float(wilcoxon(differences, alternative='two-sided').pvalue)
                    else:
                        p_value = np.nan
                    lower, upper = _bootstrap_interval(
                        differences,
                        n_bootstrap=n_bootstrap,
                        rng=rng,
                    )
                    rows.append({
                        'N_STATES': n_states,
                        'FROM_STATE': source,
                        'TO_STATE': destination,
                        'ROLLING_WINDOW': int(window),
                        'FEATURE': feature,
                        'STATISTIC': statistic,
                        'EVENT_COUNT': len(differences),
                        'EARLY_EVENT_DAYS': f'-{event_radius} to -{event_radius // 2 + 1}',
                        'BOUNDARY_EVENT_DAYS': '-2 to 0',
                        'MEDIAN_BOUNDARY_MINUS_EARLY': (
                            float(np.median(differences)) if len(differences) else np.nan
                        ),
                        'BOOTSTRAP_DIFFERENCE_CI_LOWER': lower,
                        'BOOTSTRAP_DIFFERENCE_CI_UPPER': upper,
                        'TWO_SIDED_WILCOXON_P_VALUE': p_value,
                        'RELIABLE_FOR_5_PERCENT_DECISION': reliable,
                        'NULL_HYPOTHESIS': 'rolling observable moment is unchanged near the transition boundary',
                    })

    result = pd.DataFrame(rows)
    result['FDR_ADJUSTED_P_VALUE'] = _benjamini_hochberg(result['TWO_SIDED_WILCOXON_P_VALUE'])
    result['OBSERVABLE_SHIFT_SUPPORTED_AT_5_PERCENT'] = (
        result['RELIABLE_FOR_5_PERCENT_DECISION']
        & (result['FDR_ADJUSTED_P_VALUE'] < 0.05)
    )
    return result


def _multiclass_brier(y_true, probabilities, n_states):
    encoded = np.eye(n_states)[np.asarray(y_true, dtype=int)]
    return np.mean(np.sum((encoded - probabilities) ** 2, axis=1))


def _confidence_ece(y_true, probabilities, n_bins=10):
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == np.asarray(y_true, dtype=int)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    score = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if mask.any():
            score += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(score)


def evaluate_transition_probabilities(label, split, dates, y_true, probabilities, n_states):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y_values = np.asarray(y_true, dtype=int)
    row_loss = -np.log(probabilities[np.arange(len(y_values)), y_values])
    encoded = np.eye(n_states)[y_values]
    row_brier = np.sum((encoded - probabilities) ** 2, axis=1)
    return {
        'N_STATES': n_states,
        'SPLIT': split,
        'MODEL': label,
        'SPLIT_START': pd.Index(dates).min().date(),
        'SPLIT_END': pd.Index(dates).max().date(),
        'N_OBSERVATIONS': len(y_values),
        'LOG_LOSS': float(row_loss.mean()),
        'BRIER_SCORE': float(row_brier.mean()),
        'ACCURACY': float((probabilities.argmax(axis=1) == y_values).mean()),
        'CONFIDENCE_ECE_10_BINS': _confidence_ece(y_values, probabilities),
    }, row_loss, row_brier


def _circular_block_bootstrap_interval(values, block_length, n_bootstrap, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    block_length = max(1, min(int(block_length), len(values)))
    blocks_needed = int(np.ceil(len(values) / block_length))
    statistics = []
    offsets = np.arange(block_length)
    for _ in range(int(n_bootstrap)):
        starts = rng.integers(0, len(values), size=blocks_needed)
        indices = ((starts[:, None] + offsets) % len(values)).ravel()[:len(values)]
        statistics.append(values[indices].mean())
    return tuple(np.quantile(statistics, [0.025, 0.975]))


def calibration_rows(n_states, split, label, y_true, probabilities, n_bins=10):
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)
    y_values = np.asarray(y_true, dtype=int)
    for state in range(n_states):
        predicted = probabilities[:, state]
        observed = y_values == state
        for bin_number, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            mask = (predicted >= left) & (predicted < right if right < 1 else predicted <= right)
            if not mask.any():
                continue
            rows.append({
                'N_STATES': n_states,
                'SPLIT': split,
                'MODEL': label,
                'STATE': state,
                'BIN': bin_number,
                'BIN_LEFT': left,
                'BIN_RIGHT': right,
                'COUNT': int(mask.sum()),
                'MEAN_PREDICTED_PROBABILITY': float(predicted[mask].mean()),
                'OBSERVED_FREQUENCY': float(observed[mask].mean()),
            })
    return rows


def jeffreys_smoothed_transition_matrix(states, n_states, alpha=0.5):
    """Training-only transition estimate with a Jeffreys pseudocount in every cell."""
    dates = states.index[:-1]
    train_mask = dates < TRAIN_END
    current = states.iloc[:-1].to_numpy(dtype=int)[train_mask]
    following = states.shift(-1).dropna().to_numpy(dtype=int)[train_mask]
    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (current, following), 1.0)
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_states)


def compare_abrupt_and_gradual_forecasts(
    n_states,
    states,
    posterior_probabilities,
    transition_matrix,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    block_length=DEFAULT_BLOCK_LENGTH,
    random_seed=42,
):
    hard_current = states.iloc[:-1].to_numpy(dtype=int)
    y_next = states.shift(-1).dropna().to_numpy(dtype=int)
    dates = states.index[:-1]
    matrix_variants = {
        'fixed': (transition_matrix, 0.0),
        'jeffreys_smoothed': (
            jeffreys_smoothed_transition_matrix(states, n_states),
            0.5,
        ),
    }
    reports = []
    paired = []
    calibration = []
    rng = np.random.default_rng(random_seed)

    masks = {
        'validation': (dates >= TRAIN_END) & (dates < VALIDATION_END),
        'prediction': dates >= VALIDATION_END,
    }
    for split, mask in masks.items():
        split_dates = dates[mask]
        split_y = y_next[mask]
        for variant, (matrix, pseudocount) in matrix_variants.items():
            model_rows = {}
            model_probabilities = {
                f'hard_state_{variant}_transition': matrix[hard_current][mask],
                f'soft_posterior_{variant}_transition': (
                    posterior_probabilities[:-1] @ matrix
                )[mask],
            }
            for label, probabilities in model_probabilities.items():
                report, log_losses, brier_losses = evaluate_transition_probabilities(
                    label,
                    split,
                    split_dates,
                    split_y,
                    probabilities,
                    n_states,
                )
                report['TRANSITION_MATRIX_VARIANT'] = variant
                report['DIRICHLET_PSEUDOCOUNT'] = pseudocount
                reports.append(report)
                model_rows[label] = (log_losses, brier_losses)
                calibration.extend(calibration_rows(
                    n_states,
                    split,
                    label,
                    split_y,
                    probabilities,
                ))

            hard_label = f'hard_state_{variant}_transition'
            soft_label = f'soft_posterior_{variant}_transition'
            hard_losses = model_rows[hard_label]
            soft_losses = model_rows[soft_label]
            for metric, hard_loss, soft_loss in [
                ('LOG_LOSS', hard_losses[0], soft_losses[0]),
                ('BRIER_SCORE', hard_losses[1], soft_losses[1]),
            ]:
                difference = soft_loss - hard_loss
                lower, upper = _circular_block_bootstrap_interval(
                    difference,
                    block_length,
                    n_bootstrap,
                    rng,
                )
                paired.append({
                    'N_STATES': n_states,
                    'SPLIT': split,
                    'TRANSITION_MATRIX_VARIANT': variant,
                    'DIRICHLET_PSEUDOCOUNT': pseudocount,
                    'METRIC': metric,
                    'DIFFERENCE_DEFINITION': f'{soft_label} minus {hard_label}',
                    'MEAN_PAIRED_DIFFERENCE': float(difference.mean()),
                    'BLOCK_BOOTSTRAP_CI_LOWER': lower,
                    'BLOCK_BOOTSTRAP_CI_UPPER': upper,
                    'BLOCK_LENGTH': int(block_length),
                    'BOOTSTRAP_REPLICATES': int(n_bootstrap),
                    'SOFT_POSTERIOR_SIGNIFICANTLY_BETTER': bool(upper < 0),
                    'HARD_STATE_SIGNIFICANTLY_BETTER': bool(lower > 0),
                })

    return pd.DataFrame(reports), pd.DataFrame(paired), pd.DataFrame(calibration)


def plot_transition_event(pair_summary, rolling_summary, output_path, rolling_window=20):
    source = int(pair_summary['FROM_STATE'].iloc[0])
    destination = int(pair_summary['TO_STATE'].iloc[0])
    n_states = int(pair_summary['N_STATES'].iloc[0])
    event_count = int(pair_summary['EVENT_COUNT'].max())
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

    x = pair_summary['RELATIVE_DAY'].to_numpy(dtype=float)
    axes[0].plot(x, pair_summary['MEDIAN_DESTINATION_POSTERIOR'], color='#0F6B78', linewidth=2)
    axes[0].fill_between(
        x,
        pair_summary['BOOTSTRAP_CI_LOWER'].to_numpy(dtype=float),
        pair_summary['BOOTSTRAP_CI_UPPER'].to_numpy(dtype=float),
        color='#69A9B1',
        alpha=0.3,
        label='95% event-bootstrap interval',
    )
    axes[0].set_ylabel(f'P(state {destination})')
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc='best')

    pair_rolling = rolling_summary[
        (rolling_summary['FROM_STATE'] == source)
        & (rolling_summary['TO_STATE'] == destination)
        & (rolling_summary['ROLLING_WINDOW'] == rolling_window)
    ]
    for statistic, axis, ylabel in [
        ('rolling_mean_z', axes[1], f'{rolling_window}-day mean (train z-scale)'),
        ('rolling_std_ratio', axes[2], f'{rolling_window}-day std / train std'),
    ]:
        subset = pair_rolling[pair_rolling['STATISTIC'] == statistic]
        for feature, feature_rows in subset.groupby('FEATURE'):
            feature_rows = feature_rows.sort_values('RELATIVE_DAY')
            axis.plot(feature_rows['RELATIVE_DAY'], feature_rows['MEDIAN'], label=feature)
        axis.set_ylabel(ylabel)
        axis.legend(loc='best', ncol=3, fontsize=8)

    for axis in axes:
        axis.axvline(0, color='#A63D40', linestyle='--', linewidth=1.3)
        axis.grid(alpha=0.2)
    axes[2].set_xlabel('Trading days relative to Viterbi state change')
    fig.suptitle(f'q{n_states}: state {source} to {destination} ({event_count} eligible events)')
    fig.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_calibration(calibration, output_path):
    states = sorted(calibration['STATE'].unique())
    splits = ['validation', 'prediction']
    fig, axes = plt.subplots(
        len(splits),
        len(states),
        figsize=(4 * len(states), 3.5 * len(splits)),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {'hard': '#A63D40', 'soft': '#0F6B78'}
    for row, split in enumerate(splits):
        for column, state in enumerate(states):
            axis = axes[row, column]
            subset = calibration[(calibration['SPLIT'] == split) & (calibration['STATE'] == state)]
            for label, model_rows in subset.groupby('MODEL'):
                membership = 'soft' if label.startswith('soft') else 'hard'
                matrix_variant = 'smoothed' if 'smoothed' in label else 'fixed'
                axis.plot(
                    model_rows['MEAN_PREDICTED_PROBABILITY'],
                    model_rows['OBSERVED_FREQUENCY'],
                    marker='o',
                    label=f'{membership}, {matrix_variant} matrix',
                    color=colors[membership],
                    linestyle=':' if matrix_variant == 'smoothed' else '-',
                )
            axis.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title(f'{split}: state {state}')
            axis.set_xlabel('Mean predicted probability')
            axis.set_ylabel('Observed frequency')
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    fig.suptitle('Hard-state versus soft-posterior transition calibration')
    fig.savefig(output_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def run_transition_dynamics_case(
    n_states,
    output_dir=OUTPUT_DIR,
    window_size=DEFAULT_WINDOW_SIZE,
    event_radius=DEFAULT_EVENT_RADIUS,
    rolling_windows=DEFAULT_ROLLING_WINDOWS,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    block_length=DEFAULT_BLOCK_LENGTH,
    random_seed=DEFAULT_RANDOM_SEED,
    min_events=DEFAULT_MIN_EVENTS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exog_rets = pd.read_csv(
        BASE_DIR / 'processed' / 'exog_rets.csv',
        index_col=[0],
        parse_dates=[0],
    )
    states = pd.read_csv(
        BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv',
        index_col=[0],
        parse_dates=[0],
    )['STATE'].astype(int)
    features = production_features(n_states, exog_rets)
    observations = exog_rets.loc[states.index, features].copy()
    n_train = int((observations.index < TRAIN_END).sum())
    best_params = pd.read_csv(BASE_DIR / 'phase_1' / 'best_params.csv')
    seed = int(best_params[best_params['N_STATES'] == n_states]['SEED'].iloc[0])
    model = fit_production_hmm(n_states, observations.iloc[:n_train], seed)

    reproduced = reproduce_production_states(model, observations, n_train, window_size)
    match_share = float(np.mean(reproduced == states.to_numpy(dtype=int)))
    if match_share < 1.0:
        raise RuntimeError(
            f'q{n_states} production-state reconstruction matched only {match_share:.2%} of saved labels.'
        )

    probabilities = windowed_filtered_probabilities(model, observations, window_size)
    posterior = posterior_frame(states.index, states, probabilities)
    moments = rolling_moment_frame(observations, n_train, rolling_windows)
    events, counts = build_event_observations(
        n_states,
        states,
        posterior,
        moments,
        event_radius,
    )
    posterior_summary = posterior_event_summary(events, n_bootstrap, random_seed + n_states)
    rolling_summary = rolling_event_summary(events, rolling_windows)
    monotonicity = transition_monotonicity_tests(
        events,
        event_radius,
        min_events,
        n_bootstrap,
        random_seed + 10 * n_states,
    )
    observable_tests = observable_shift_tests(
        events,
        rolling_windows,
        event_radius,
        min_events,
        n_bootstrap,
        random_seed + 20 * n_states,
    )
    transition_matrix = pd.read_csv(
        BASE_DIR / 'phase_1' / f'hmm_trans_mat_q{n_states}.csv',
        index_col=[0],
    ).to_numpy(dtype=float)
    comparison, paired, calibration = compare_abrupt_and_gradual_forecasts(
        n_states,
        states,
        probabilities,
        transition_matrix,
        n_bootstrap,
        block_length,
        random_seed + 30 * n_states,
    )

    paths = {
        'posterior': output_dir / f'filtered_state_probabilities_q{n_states}.csv',
        'events': output_dir / f'transition_event_observations_q{n_states}.csv',
        'counts': output_dir / f'transition_event_counts_q{n_states}.csv',
        'posterior_summary': output_dir / f'transition_posterior_event_summary_q{n_states}.csv',
        'rolling_summary': output_dir / f'transition_rolling_moment_summary_q{n_states}.csv',
        'monotonicity': output_dir / f'transition_monotonicity_tests_q{n_states}.csv',
        'observable_tests': output_dir / f'transition_observable_shift_tests_q{n_states}.csv',
        'comparison': output_dir / f'transition_timing_model_comparison_q{n_states}.csv',
        'paired': output_dir / f'transition_timing_paired_tests_q{n_states}.csv',
        'calibration': output_dir / f'transition_calibration_bins_q{n_states}.csv',
        'calibration_plot': output_dir / f'transition_calibration_q{n_states}.png',
    }
    posterior.to_csv(paths['posterior'])
    events.to_csv(paths['events'], index=False)
    counts.to_csv(paths['counts'], index=False)
    posterior_summary.to_csv(paths['posterior_summary'], index=False)
    rolling_summary.to_csv(paths['rolling_summary'], index=False)
    monotonicity.to_csv(paths['monotonicity'], index=False)
    observable_tests.to_csv(paths['observable_tests'], index=False)
    comparison.to_csv(paths['comparison'], index=False)
    paired.to_csv(paths['paired'], index=False)
    calibration.to_csv(paths['calibration'], index=False)
    plot_calibration(calibration, paths['calibration_plot'])

    event_plot_paths = []
    for row in counts[counts['ELIGIBLE_EVENTS'] >= min_events].itertuples():
        pair = posterior_summary[
            (posterior_summary['FROM_STATE'] == row.FROM_STATE)
            & (posterior_summary['TO_STATE'] == row.TO_STATE)
        ].sort_values('RELATIVE_DAY')
        path = output_dir / (
            f'transition_event_q{n_states}_from_{row.FROM_STATE}_to_{row.TO_STATE}.png'
        )
        plot_transition_event(pair, rolling_summary, path, rolling_window=max(rolling_windows))
        event_plot_paths.append(path)

    config = pd.DataFrame([{
        'N_STATES': n_states,
        'FEATURES': '|'.join(features),
        'TRAIN_END_EXCLUSIVE': TRAIN_END,
        'VALIDATION_END_EXCLUSIVE': VALIDATION_END,
        'PRODUCTION_SEED': seed,
        'PRODUCTION_LABEL_MATCH_SHARE': match_share,
        'FILTER_TYPE': 'one-sided posterior reset on each trailing production window',
        'WINDOW_SIZE': int(window_size),
        'EVENT_RADIUS': int(event_radius),
        'ROLLING_WINDOWS': '|'.join(str(value) for value in rolling_windows),
        'ROLLING_MOMENT_STANDARDIZATION': 'training mean and population standard deviation',
        'BOOTSTRAP_REPLICATES': int(n_bootstrap),
        'PAIRED_TEST_BLOCK_LENGTH': int(block_length),
        'MIN_EVENTS_FOR_INFERENCE': int(min_events),
        'RANDOM_SEED': int(random_seed),
        'EXISTING_CALIBRATION_MODIFIED': False,
    }])
    config_path = output_dir / f'transition_dynamics_run_config_q{n_states}.csv'
    config.to_csv(config_path, index=False)
    paths['config'] = config_path
    paths['event_plots'] = event_plot_paths
    return {
        'posterior': posterior,
        'events': events,
        'counts': counts,
        'posterior_summary': posterior_summary,
        'rolling_summary': rolling_summary,
        'monotonicity': monotonicity,
        'observable_tests': observable_tests,
        'comparison': comparison,
        'paired': paired,
        'calibration': calibration,
        'paths': paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, nargs='+', default=list(DEFAULT_STATES))
    parser.add_argument('--bootstrap-replicates', type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument('--event-radius', type=int, default=DEFAULT_EVENT_RADIUS)
    parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE)
    args = parser.parse_args()

    generated = []
    for n_states in args.states:
        result = run_transition_dynamics_case(
            n_states,
            window_size=args.window_size,
            event_radius=args.event_radius,
            n_bootstrap=args.bootstrap_replicates,
        )
        generated.extend(
            path for key, path in result['paths'].items()
            if key != 'event_plots'
        )
        generated.extend(result['paths']['event_plots'])
        print(f'Completed q{n_states} transition-dynamics diagnostics.')

    print(f'Outputs saved under: {OUTPUT_DIR}')
    for path in generated:
        print(f'- {path}')


if __name__ == '__main__':
    main()
