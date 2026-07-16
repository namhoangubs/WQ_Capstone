from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from numpy import arange, inf, isnan, nan, zeros
from pandas import DataFrame, cut, read_csv

from p1_3_transition_features import build_transition_dataset
from p1_4_dynamic_transition import (
    TRAIN_END,
    VALIDATION_END,
    _align_probabilities,
    fit_dynamic_transition_model,
    save_dynamic_transition_report,
    split_transition_dataset,
)
from project_config import BASE_DIR


OUTPUT_DIR = BASE_DIR / 'phase_1' / 'transition_comparison'


def discover_hmm_cases():
    cases = []
    for path in (BASE_DIR / 'phase_1').glob('hmm_trans_mat_q*.csv'):
        match = re.search(r'q(\d+)', path.stem)
        if match:
            cases.append(int(match.group(1)))

    return sorted(cases)


def read_fixed_transition_matrix(n_states):
    path = BASE_DIR / 'phase_1' / f'hmm_trans_mat_q{n_states}.csv'
    return read_csv(path, index_col=[0]).astype(float)


def read_stock_states(n_states):
    path = BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv'
    stock_rets = read_csv(path, index_col=[0], parse_dates=[0])
    return stock_rets['STATE'].astype(int)


def observed_transition_matrix(current_states, next_states, state_classes):
    matrix = zeros((len(state_classes), len(state_classes)))

    for row, current_state in enumerate(state_classes):
        mask = current_states == current_state
        count = mask.sum()

        if count == 0:
            matrix[row, :] = nan
            continue

        next_for_state = next_states[mask]
        for column, next_state in enumerate(state_classes):
            matrix[row, column] = (next_for_state == next_state).mean()

    return matrix


def average_probability_matrix(current_states, probabilities, state_classes):
    matrix = zeros((len(state_classes), len(state_classes)))

    for row, current_state in enumerate(state_classes):
        mask = current_states == current_state
        if mask.sum() == 0:
            matrix[row, :] = nan
        else:
            matrix[row, :] = probabilities[mask].mean(axis=0)

    return matrix


def prediction_window_observed_matrix(n_states):
    states = read_stock_states(n_states)
    current_states = states.iloc[:-1]
    next_states = states.shift(-1).dropna().astype(int)
    prediction_mask = current_states.index >= VALIDATION_END
    current_prediction = current_states[prediction_mask].to_numpy(dtype=int)
    next_prediction = next_states[prediction_mask].to_numpy(dtype=int)
    state_classes = sorted(states.dropna().unique())

    return observed_transition_matrix(current_prediction, next_prediction, state_classes)


def load_or_create_dynamic_report(n_states):
    path = BASE_DIR / 'phase_1' / f'dynamic_transition_report_q{n_states}.csv'
    if path.exists():
        report = read_csv(path)
        if 'SPLIT' in report.columns:
            return report

    report, _ = save_dynamic_transition_report(n_states)
    return report


def best_dynamic_lag_config(n_states):
    report = load_or_create_dynamic_report(n_states)
    dynamic_rows = report[
        (report['MODEL'] == 'dynamic_duration_logit')
        & (report['SPLIT'] == 'validation')
    ].copy()

    if dynamic_rows.empty:
        raise ValueError(f'No dynamic transition validation rows found for q{n_states}.')

    return dynamic_rows.sort_values(
        ['LOG_LOSS', 'BRIER_SCORE', 'STATE_LAG', 'EXOG_LAG'],
        ascending=[True, True, True, True],
    ).iloc[0]


def dynamic_comparison_data(n_states):
    best_config = best_dynamic_lag_config(n_states)
    state_lag = int(best_config['STATE_LAG'])
    exog_lag = int(best_config['EXOG_LAG'])

    stock_states = read_stock_states(n_states)
    exog_rets = read_csv(BASE_DIR / 'processed' / 'exog_rets.csv', index_col=[0], parse_dates=[0])
    fixed_matrix = read_fixed_transition_matrix(n_states)

    X, y, meta, state_classes = build_transition_dataset(
        stock_states,
        exog_rets,
        max_state_lag=state_lag,
        max_exog_lag=exog_lag,
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']
    X_prediction, y_prediction, meta_prediction = splits['prediction']

    model = fit_dynamic_transition_model(X_train, y_train)
    logit = model.named_steps['logit']
    dynamic_probabilities = _align_probabilities(
        model.predict_proba(X_prediction),
        logit.classes_,
        state_classes,
    )

    current_prediction = meta_prediction['CURRENT_STATE'].to_numpy(dtype=int)
    y_prediction_values = y_prediction.to_numpy(dtype=int)
    observed_matrix = observed_transition_matrix(current_prediction, y_prediction_values, state_classes)
    dynamic_matrix = average_probability_matrix(current_prediction, dynamic_probabilities, state_classes)

    return {
        'n_states': n_states,
        'state_lag': state_lag,
        'exog_lag': exog_lag,
        'state_classes': state_classes,
        'fixed_matrix': fixed_matrix.to_numpy(dtype=float),
        'observed_matrix': observed_matrix,
        'dynamic_matrix': dynamic_matrix,
        'X_prediction': X_prediction,
        'meta_prediction': meta_prediction,
        'dynamic_probabilities': dynamic_probabilities,
        'prediction_start': X_prediction.index.min().date(),
        'prediction_end': X_prediction.index.max().date(),
    }


def matrix_records(n_states, state_classes, fixed_matrix, observed_matrix, dynamic_matrix=None):
    rows = []

    for row, from_state in enumerate(state_classes):
        for column, to_state in enumerate(state_classes):
            fixed_value = fixed_matrix[row, column]
            observed_value = observed_matrix[row, column]
            dynamic_value = None
            diff_value = None

            if dynamic_matrix is not None:
                dynamic_value = dynamic_matrix[row, column]
                diff_value = dynamic_value - fixed_value

            rows.append({
                'N_STATES': n_states,
                'FROM_STATE': from_state,
                'TO_STATE': to_state,
                'OBSERVED_PREDICTION': observed_value,
                'FIXED_HMM': fixed_value,
                'DYNAMIC_AVERAGE': dynamic_value,
                'DYNAMIC_MINUS_FIXED': diff_value,
            })

    return rows


def annotate_heatmap(ax, matrix, diverging=False):
    finite_values = matrix[~isnan(matrix)]
    threshold = 0 if diverging else 0.5

    if not diverging and len(finite_values) > 0:
        threshold = finite_values.max() / 2

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if isnan(value):
                label = 'NA'
                color = 'black'
            else:
                label = f'{value:.3f}'
                color = 'white' if abs(value) > threshold else 'black'

            ax.text(column, row, label, ha='center', va='center', color=color, fontsize=9)


def draw_heatmap(ax, matrix, title, state_classes, cmap='Blues', vmin=0, vmax=1, norm=None):
    if norm is None:
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        image = ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel('Next state')
    ax.set_ylabel('Current state')
    ax.set_xticks(arange(len(state_classes)))
    ax.set_yticks(arange(len(state_classes)))
    ax.set_xticklabels(state_classes)
    ax.set_yticklabels(state_classes)
    annotate_heatmap(ax, matrix, diverging=norm is not None)
    return image


def plot_fixed_only_case(n_states, output_dir):
    state_classes = sorted(read_stock_states(n_states).dropna().unique())
    fixed_matrix = read_fixed_transition_matrix(n_states).to_numpy(dtype=float)
    observed_matrix = prediction_window_observed_matrix(n_states)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    image = draw_heatmap(axes[0], observed_matrix, 'Observed (prediction window)', state_classes)
    draw_heatmap(axes[1], fixed_matrix, f'Fixed HMM (trained to {TRAIN_END})', state_classes)
    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    fig.suptitle(f'q{n_states}: fixed transition matrix only (prediction window from {VALIDATION_END})')

    path = output_dir / f'transition_matrix_fixed_only_q{n_states}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return matrix_records(n_states, state_classes, fixed_matrix, observed_matrix)


def plot_matrix_comparison(comparison, output_dir):
    n_states = comparison['n_states']
    state_classes = comparison['state_classes']
    fixed_matrix = comparison['fixed_matrix']
    observed_matrix = comparison['observed_matrix']
    dynamic_matrix = comparison['dynamic_matrix']
    diff_matrix = dynamic_matrix - fixed_matrix

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    image = draw_heatmap(axes[0], observed_matrix, 'Observed (prediction window)', state_classes)
    draw_heatmap(axes[1], fixed_matrix, f'Fixed HMM (trained to {TRAIN_END})', state_classes)
    draw_heatmap(axes[2], dynamic_matrix, f'Dynamic average (trained to {TRAIN_END})', state_classes)

    max_abs_diff = max(0.001, abs(diff_matrix[~isnan(diff_matrix)]).max())
    norm = TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0, vmax=max_abs_diff)
    diff_image = draw_heatmap(
        axes[3],
        diff_matrix,
        'Dynamic - fixed',
        state_classes,
        cmap='RdBu_r',
        norm=norm,
    )

    fig.colorbar(image, ax=axes[:3], fraction=0.025, pad=0.02)
    fig.colorbar(diff_image, ax=axes[3], fraction=0.045, pad=0.04)
    fig.suptitle(
        f"q{n_states}: transition matrices on prediction window "
        f"{comparison['prediction_start']} to {comparison['prediction_end']} "
        f"(lags selected on validation: state={comparison['state_lag']}, exog={comparison['exog_lag']})"
    )

    path = output_dir / f'transition_matrix_comparison_q{n_states}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return matrix_records(n_states, state_classes, fixed_matrix, observed_matrix, dynamic_matrix)


def plot_duration_stay_probability(comparison, output_dir):
    n_states = comparison['n_states']
    state_classes = comparison['state_classes']
    fixed_matrix = comparison['fixed_matrix']
    X_prediction = comparison['X_prediction']
    meta_prediction = comparison['meta_prediction']
    dynamic_probabilities = comparison['dynamic_probabilities']

    duration_bins = [0, 1, 2, 5, 10, 20, 60, inf]
    duration_labels = ['1', '2', '3-5', '6-10', '11-20', '21-60', '61+']
    duration_bucket = cut(X_prediction['DURATION'], bins=duration_bins, labels=duration_labels)

    n_columns = min(2, len(state_classes))
    n_rows = int((len(state_classes) + n_columns - 1) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(6.5 * n_columns, 3.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()

    for index, current_state in enumerate(state_classes):
        ax = axes[index]
        mask = meta_prediction['CURRENT_STATE'].to_numpy(dtype=int) == current_state
        stay_column = state_classes.index(current_state)
        stay_probabilities = dynamic_probabilities[:, stay_column]

        frame = DataFrame({
            'DURATION_BUCKET': duration_bucket[mask],
            'STAY_PROBABILITY': stay_probabilities[mask],
        }).dropna()

        grouped = frame.groupby('DURATION_BUCKET', observed=False)['STAY_PROBABILITY'].mean()
        grouped = grouped.reindex(duration_labels)

        ax.plot(duration_labels, grouped.values, marker='o', label='Dynamic')
        ax.axhline(
            fixed_matrix[index, index],
            color='black',
            linestyle='--',
            linewidth=1.2,
            label='Fixed HMM',
        )
        ax.set_title(f'Current state {current_state}')
        ax.set_xlabel('Consecutive days already in state')
        ax.set_ylabel('P(stay in same state)')
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis='x', rotation=35)
        ax.legend(loc='best')

    for index in range(len(state_classes), len(axes)):
        axes[index].axis('off')

    fig.suptitle(f'q{n_states}: duration-dependent stay probability (prediction window)')
    path = output_dir / f'duration_stay_probability_q{n_states}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_metric_delta_grids(cases, output_dir, metric, split):
    dynamic_cases = [case for case in cases if case > 1]
    if not dynamic_cases:
        return None

    fig, axes = plt.subplots(1, len(dynamic_cases), figsize=(6 * len(dynamic_cases), 4.2), constrained_layout=True)
    if len(dynamic_cases) == 1:
        axes = [axes]

    for ax, n_states in zip(axes, dynamic_cases):
        report = load_or_create_dynamic_report(n_states)
        report = report[report['SPLIT'] == split]
        fixed = report[report['MODEL'] == 'fixed_hmm_transition']
        dynamic = report[report['MODEL'] == 'dynamic_duration_logit']
        merged = fixed.merge(
            dynamic,
            on=['N_STATES', 'STATE_LAG', 'EXOG_LAG', 'TRAIN_ROWS', 'EVAL_ROWS'],
            suffixes=('_FIXED', '_DYNAMIC'),
        )
        merged['DELTA'] = merged[f'{metric}_DYNAMIC'] - merged[f'{metric}_FIXED']
        grid = merged.pivot(index='STATE_LAG', columns='EXOG_LAG', values='DELTA')
        max_abs = max(0.001, abs(grid.to_numpy()).max())
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        image = ax.imshow(grid, cmap='RdYlGn_r', norm=norm)
        ax.set_title(f'q{n_states}')
        ax.set_xlabel('Exogenous lag')
        ax.set_ylabel('State lag')
        ax.set_xticks(arange(len(grid.columns)))
        ax.set_yticks(arange(len(grid.index)))
        ax.set_xticklabels(grid.columns)
        ax.set_yticklabels(grid.index)
        annotate_heatmap(ax, grid.to_numpy(), diverging=True)

    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    fig.suptitle(f'Dynamic minus fixed {metric.lower()} by lag setting on the {split} split; negative is better')

    path = output_dir / f'transition_{metric.lower()}_delta_grid_{split}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = discover_hmm_cases()
    all_records = []
    generated_graphs = []

    for n_states in cases:
        if n_states <= 1:
            all_records.extend(plot_fixed_only_case(n_states, OUTPUT_DIR))
            generated_graphs.append(OUTPUT_DIR / f'transition_matrix_fixed_only_q{n_states}.png')
            continue

        comparison = dynamic_comparison_data(n_states)
        all_records.extend(plot_matrix_comparison(comparison, OUTPUT_DIR))
        plot_duration_stay_probability(comparison, OUTPUT_DIR)
        generated_graphs.extend([
            OUTPUT_DIR / f'transition_matrix_comparison_q{n_states}.png',
            OUTPUT_DIR / f'duration_stay_probability_q{n_states}.png',
        ])

    summary = DataFrame(all_records)
    summary_path = OUTPUT_DIR / 'transition_matrix_summary.csv'
    summary.to_csv(summary_path, index=False)

    for metric in ['LOG_LOSS', 'BRIER_SCORE']:
        for split in ['validation', 'prediction']:
            path = plot_metric_delta_grids(cases, OUTPUT_DIR, metric, split)
            if path is not None:
                generated_graphs.append(path)

    print(f'Train window: start of data to {TRAIN_END} (matches the fixed HMM training window)')
    print(f'Validation window (lag selection): {TRAIN_END} to {VALIDATION_END}')
    print(f'Prediction window (final comparison): {VALIDATION_END} onward')
    print(f'Summary table saved to: {summary_path}')
    print('Graphs saved:')
    for path in generated_graphs:
        print(f'- {path}')


if __name__ == '__main__':
    main()
