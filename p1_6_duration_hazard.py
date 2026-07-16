import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import clip, exp, inf, log, outer, zeros
from pandas import DataFrame, concat, cut, read_csv
from scipy.optimize import minimize_scalar

from p1_3_transition_features import build_transition_dataset
from p1_4_dynamic_transition import (
    TRAIN_END,
    VALIDATION_END,
    evaluate_probabilities,
    fixed_transition_probabilities,
    split_transition_dataset,
)
from p1_5_transition_comparison import (
    OUTPUT_DIR,
    average_probability_matrix,
    discover_hmm_cases,
    draw_heatmap,
    matrix_records,
    observed_transition_matrix,
    read_fixed_transition_matrix,
)
from project_config import BASE_DIR

# The scheme keeps the fixed HMM transition matrix as the base and adds a
# duration hazard on the diagonal:
#   logit(P_stay(i, d)) = logit(P_ii_fixed) - beta_i * ln(d)
# With beta_i >= 0 the probability of leaving a regime increases the longer
# the process has already stayed in it. beta_i = 0 recovers the fixed HMM and
# at duration 1 the scheme always equals the fixed matrix. Exit mass follows
# the fixed matrix off-diagonal proportions.
CONSTRAINED_BETA_BOUNDS = (0.0, 10.0)
UNCONSTRAINED_BETA_BOUNDS = (-10.0, 10.0)


def _logit(probability):
    probability = clip(probability, 1e-12, 1 - 1e-12)
    return log(probability / (1 - probability))


def _sigmoid(value):
    return 1 / (1 + exp(-value))


def stay_probability(base_stay, durations, beta):
    return _sigmoid(_logit(base_stay) - beta * log(durations))


def fit_state_beta(durations, stayed, base_stay, bounds):
    def negative_log_likelihood(beta):
        p_stay = clip(stay_probability(base_stay, durations, beta), 1e-12, 1 - 1e-12)
        return -(log(p_stay[stayed]).sum() + log(1 - p_stay[~stayed]).sum())

    result = minimize_scalar(negative_log_likelihood, bounds=bounds, method='bounded')
    return float(result.x)


def fit_duration_hazard(current_states, durations, next_states, transition_matrix, state_classes,
                        bounds=CONSTRAINED_BETA_BOUNDS):
    matrix = transition_matrix.to_numpy(dtype=float)
    betas = {}

    for row_index, state in enumerate(state_classes):
        mask = current_states == state
        betas[state] = fit_state_beta(
            durations[mask],
            next_states[mask] == state,
            matrix[row_index, row_index],
            bounds,
        )

    return betas


def duration_hazard_probabilities(current_states, durations, transition_matrix, state_classes, betas):
    matrix = transition_matrix.to_numpy(dtype=float)
    probabilities = zeros((len(current_states), len(state_classes)))

    for row_index, state in enumerate(state_classes):
        mask = current_states == state
        if not mask.any():
            continue

        base_stay = matrix[row_index, row_index]
        p_stay = stay_probability(base_stay, durations[mask], betas[state])

        exit_weights = matrix[row_index].copy()
        exit_weights[row_index] = 0.0
        total_exit = exit_weights.sum()
        if total_exit > 0:
            exit_weights = exit_weights / total_exit
        else:
            exit_weights[:] = 1.0 / max(1, len(state_classes) - 1)
            exit_weights[row_index] = 0.0

        probabilities[mask] = outer(1 - p_stay, exit_weights)
        probabilities[mask, row_index] = p_stay

    return probabilities


def evaluate_duration_hazard(n_states):
    stock_rets = read_csv(BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv', index_col=[0], parse_dates=[0])
    exog_rets = read_csv(BASE_DIR / 'processed' / 'exog_rets.csv', index_col=[0], parse_dates=[0])
    transition_matrix = read_fixed_transition_matrix(n_states)

    X, y, meta, state_classes = build_transition_dataset(
        stock_rets['STATE'],
        exog_rets,
        max_state_lag=0,
        max_exog_lag=0,
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, meta_train = splits['train']

    train_current = meta_train['CURRENT_STATE'].to_numpy(dtype=int)
    train_durations = X_train['DURATION'].to_numpy(dtype=float)
    train_next = y_train.to_numpy(dtype=int)

    betas = fit_duration_hazard(
        train_current, train_durations, train_next, transition_matrix, state_classes
    )
    diagnostic_betas = fit_duration_hazard(
        train_current, train_durations, train_next, transition_matrix, state_classes,
        bounds=UNCONSTRAINED_BETA_BOUNDS,
    )

    beta_rows = []
    for state in state_classes:
        beta_rows.append({
            'N_STATES': n_states,
            'STATE': state,
            'TRAIN_ROWS_STATE': int((train_current == state).sum()),
            'BETA_CONSTRAINED': betas[state],
            'BETA_UNCONSTRAINED': diagnostic_betas[state],
        })

    report_rows = []
    evaluation = {}
    for split_name in ['validation', 'prediction']:
        X_eval, y_eval, meta_eval = splits[split_name]
        current_eval = meta_eval['CURRENT_STATE'].to_numpy(dtype=int)
        durations_eval = X_eval['DURATION'].to_numpy(dtype=float)
        y_eval_values = y_eval.to_numpy(dtype=int)

        extra = {
            'SPLIT': split_name,
            'SPLIT_START': X_eval.index.min().date(),
            'SPLIT_END': X_eval.index.max().date(),
            'N_STATES': n_states,
            'TRAIN_ROWS': len(X_train),
            'EVAL_ROWS': len(X_eval),
        }

        fixed_probs = fixed_transition_probabilities(
            meta_eval['CURRENT_STATE'], transition_matrix, state_classes
        )
        report_rows.append(evaluate_probabilities(
            'fixed_hmm_transition', y_eval_values, fixed_probs, state_classes, extra
        ))

        hazard_probs = duration_hazard_probabilities(
            current_eval, durations_eval, transition_matrix, state_classes, betas
        )
        report_rows.append(evaluate_probabilities(
            'duration_hazard_hmm', y_eval_values, hazard_probs, state_classes, extra
        ))

        evaluation[split_name] = {
            'current': current_eval,
            'durations': durations_eval,
            'y': y_eval_values,
            'hazard_probabilities': hazard_probs,
            'start': X_eval.index.min().date(),
            'end': X_eval.index.max().date(),
        }

    return {
        'n_states': n_states,
        'state_classes': state_classes,
        'transition_matrix': transition_matrix.to_numpy(dtype=float),
        'betas': betas,
        'beta_rows': beta_rows,
        'report': DataFrame(report_rows),
        'evaluation': evaluation,
    }


def plot_hazard_matrix_comparison(result, output_dir):
    n_states = result['n_states']
    state_classes = result['state_classes']
    fixed_matrix = result['transition_matrix']
    prediction = result['evaluation']['prediction']

    observed_matrix = observed_transition_matrix(prediction['current'], prediction['y'], state_classes)
    hazard_matrix = average_probability_matrix(prediction['current'], prediction['hazard_probabilities'], state_classes)

    from matplotlib.colors import TwoSlopeNorm
    from numpy import isnan

    diff_matrix = hazard_matrix - fixed_matrix

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    image = draw_heatmap(axes[0], observed_matrix, 'Observed (prediction window)', state_classes)
    draw_heatmap(axes[1], fixed_matrix, f'Fixed HMM (trained to {TRAIN_END})', state_classes)
    draw_heatmap(axes[2], hazard_matrix, 'Duration hazard average', state_classes)

    max_abs_diff = max(0.001, abs(diff_matrix[~isnan(diff_matrix)]).max())
    norm = TwoSlopeNorm(vmin=-max_abs_diff, vcenter=0, vmax=max_abs_diff)
    diff_image = draw_heatmap(axes[3], diff_matrix, 'Hazard - fixed', state_classes, cmap='RdBu_r', norm=norm)

    fig.colorbar(image, ax=axes[:3], fraction=0.025, pad=0.02)
    fig.colorbar(diff_image, ax=axes[3], fraction=0.045, pad=0.04)
    fig.suptitle(
        f"q{n_states}: duration-hazard scheme on prediction window "
        f"{prediction['start']} to {prediction['end']}"
    )

    path = output_dir / f'transition_matrix_hazard_q{n_states}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return path, matrix_records(n_states, state_classes, fixed_matrix, observed_matrix, hazard_matrix)


def plot_hazard_stay_curves(result, output_dir):
    n_states = result['n_states']
    state_classes = result['state_classes']
    fixed_matrix = result['transition_matrix']
    betas = result['betas']
    prediction = result['evaluation']['prediction']

    duration_bins = [0, 1, 2, 5, 10, 20, 60, inf]
    duration_labels = ['1', '2', '3-5', '6-10', '11-20', '21-60', '61+']
    duration_bucket = cut(prediction['durations'], bins=duration_bins, labels=duration_labels)

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
        mask = prediction['current'] == current_state
        stayed = (prediction['y'] == current_state).astype(float)
        model_stay = prediction['hazard_probabilities'][:, state_classes.index(current_state)]

        frame = DataFrame({
            'DURATION_BUCKET': duration_bucket[mask],
            'EMPIRICAL_STAY': stayed[mask],
            'HAZARD_STAY': model_stay[mask],
        }).dropna()

        grouped = frame.groupby('DURATION_BUCKET', observed=False).mean()
        grouped = grouped.reindex(duration_labels)

        ax.plot(duration_labels, grouped['EMPIRICAL_STAY'].values, marker='s', color='tab:grey',
                label='Observed (prediction window)')
        ax.plot(duration_labels, grouped['HAZARD_STAY'].values, marker='o',
                label=f'Duration hazard (beta={betas[current_state]:.3f})')
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
        ax.legend(loc='best', fontsize=8)

    for index in range(len(state_classes), len(axes)):
        axes[index].axis('off')

    fig.suptitle(f'q{n_states}: duration-hazard stay probability (beta >= 0 forces exit to rise with duration)')
    path = output_dir / f'duration_hazard_stay_q{n_states}.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Train window (fit betas and fixed HMM): start of data to {TRAIN_END}')
    print(f'Validation window: {TRAIN_END} to {VALIDATION_END}')
    print(f'Prediction window (final comparison): {VALIDATION_END} onward')

    cases = [case for case in discover_hmm_cases() if case > 1]
    all_reports = []
    all_beta_rows = []
    all_records = []
    generated = []

    for n_states in cases:
        result = evaluate_duration_hazard(n_states)
        all_reports.append(result['report'])
        all_beta_rows.extend(result['beta_rows'])

        report_path = BASE_DIR / 'phase_1' / f'duration_hazard_report_q{n_states}.csv'
        result['report'].to_csv(report_path, index=False)

        heatmap_path, records = plot_hazard_matrix_comparison(result, OUTPUT_DIR)
        curve_path = plot_hazard_stay_curves(result, OUTPUT_DIR)
        all_records.extend(records)
        generated.extend([report_path, heatmap_path, curve_path])

        print(f'\nN_STATES={n_states}')
        print('Fitted betas (constrained to >= 0; unconstrained shown as diagnostic):')
        print(DataFrame(result['beta_rows']).to_string(index=False))
        print('Fixed HMM versus duration-hazard scheme:')
        print(result['report'].to_string(index=False))

    betas_path = BASE_DIR / 'phase_1' / 'duration_hazard_betas.csv'
    DataFrame(all_beta_rows).to_csv(betas_path, index=False)

    summary = DataFrame(all_records)
    summary_path = OUTPUT_DIR / 'duration_hazard_matrix_summary.csv'
    summary.to_csv(summary_path, index=False)

    print(f'\nBetas saved to: {betas_path}')
    print(f'Matrix summary saved to: {summary_path}')
    print('Outputs saved:')
    for path in generated:
        print(f'- {path}')


if __name__ == '__main__':
    main()
