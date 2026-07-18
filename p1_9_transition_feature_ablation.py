"""Feature-group ablation and conditional permutation diagnostics for MLR/MLG."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p1_3_transition_features import build_transition_dataset
from p1_4_dynamic_transition import (
    _align_probabilities,
    multiclass_brier_score,
    split_transition_dataset,
)
from p1_4_mlg_transition import (
    MLGFeatureTransformer,
    MLGTransitionModel,
    PreparedMLGDesign,
    best_mlg_config,
)
from project_config import BASE_DIR


OUTPUT_DIR = BASE_DIR / 'phase_1' / 'transition_feature_ablation'
DEFAULT_STATES = (2, 4)
DEFAULT_PERMUTATIONS = 199
DEFAULT_BLOCK_LENGTH = 20
DEFAULT_RANDOM_SEED = 42
CANDIDATE_ORDER = (
    'state_only',
    'state_market',
    'state_duration',
    'state_market_restricted_duration',
    'full',
)


def feature_group(column):
    """Map a raw transition feature to state, market, or duration."""
    if column.startswith('STATE_LAG_'):
        return 'state'
    if column.startswith('DURATION'):
        return 'duration'
    return 'market'


def grouped_columns(columns):
    groups = {'state': [], 'market': [], 'duration': []}
    for column in columns:
        groups[feature_group(column)].append(column)
    return groups


def multiclass_expected_calibration_error(y_true, probabilities, state_classes, n_bins=10):
    """Confidence-based expected calibration error for a multiclass forecast."""
    classes = np.asarray(state_classes)
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    correct = predicted == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bins = np.minimum(np.digitize(confidence, edges[1:-1], right=True), n_bins - 1)

    ece = 0.0
    for bin_index in range(int(n_bins)):
        mask = bins == bin_index
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def probability_metrics(y_true, current_states, probabilities, state_classes):
    """Return multiclass and stay-versus-exit probability diagnostics."""
    y_values = np.asarray(y_true, dtype=int)
    current_values = np.asarray(current_states, dtype=int)
    classes = np.asarray(state_classes, dtype=int)
    class_to_column = {state: index for index, state in enumerate(classes)}
    current_columns = np.array([class_to_column[state] for state in current_values])
    predicted = classes[np.argmax(probabilities, axis=1)]

    exit_actual = (y_values != current_values).astype(int)
    exit_probability = 1.0 - probabilities[np.arange(len(probabilities)), current_columns]
    exit_probability = np.clip(exit_probability, 1e-12, 1.0 - 1e-12)
    exit_predicted = exit_probability >= 0.5

    row = {
        'LOG_LOSS': log_loss(y_values, probabilities, labels=classes),
        'BRIER_SCORE': multiclass_brier_score(y_values, probabilities, classes),
        'ACCURACY': accuracy_score(y_values, predicted),
        'ECE': multiclass_expected_calibration_error(y_values, probabilities, classes),
        'EXIT_LOG_LOSS': log_loss(exit_actual, exit_probability, labels=[0, 1]),
        'EXIT_BRIER_SCORE': float(np.mean((exit_actual - exit_probability) ** 2)),
        'EXIT_RATE_ACTUAL': float(exit_actual.mean()),
        'EXIT_RATE_PREDICTED': float(exit_probability.mean()),
        'EXIT_RECALL_AT_0_5': recall_score(exit_actual, exit_predicted, zero_division=0),
        'EXIT_PRECISION_AT_0_5': precision_score(exit_actual, exit_predicted, zero_division=0),
    }
    if np.unique(exit_actual).size == 2:
        row['EXIT_ROC_AUC'] = roc_auc_score(exit_actual, exit_probability)
        row['EXIT_AVERAGE_PRECISION'] = average_precision_score(exit_actual, exit_probability)
    else:
        row['EXIT_ROC_AUC'] = np.nan
        row['EXIT_AVERAGE_PRECISION'] = np.nan
    return row


def state_conditional_block_permutation(X, columns, current_states, block_length, rng):
    """Jointly shuffle a feature group in blocks, separately within each current state."""
    permuted = X.copy()
    if not columns:
        return permuted

    current_values = np.asarray(current_states, dtype=int)
    for state in np.unique(current_values):
        positions = np.flatnonzero(current_values == state)
        if len(positions) < 2:
            continue
        local_positions = np.arange(len(positions))
        blocks = [
            local_positions[start:start + int(block_length)]
            for start in range(0, len(local_positions), int(block_length))
        ]
        source_order = np.concatenate([blocks[index] for index in rng.permutation(len(blocks))])
        values = X.iloc[positions][columns].to_numpy(copy=True)
        permuted.iloc[positions, permuted.columns.get_indexer(columns)] = values[source_order]
    return permuted


def _read_states_and_exog(n_states):
    stock = pd.read_csv(
        BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv',
        index_col=0,
        parse_dates=[0],
    )
    exog = pd.read_csv(
        BASE_DIR / 'processed' / 'exog_rets.csv',
        index_col=0,
        parse_dates=[0],
    )
    return stock['STATE'].astype(int), exog


def _best_mlr_config(n_states):
    report = pd.read_csv(BASE_DIR / 'phase_1' / f'dynamic_transition_report_q{n_states}.csv')
    candidates = report[
        (report['MODEL'] == 'dynamic_duration_logit')
        & (report['SPLIT'] == 'validation')
    ].copy()
    if candidates.empty:
        raise ValueError(f'No selected MLR validation candidate is available for q{n_states}.')
    return candidates.sort_values(
        ['LOG_LOSS', 'BRIER_SCORE', 'ACCURACY', 'STATE_LAG', 'EXOG_LAG'],
        ascending=[True, True, False, True, True],
    ).iloc[0]


def _load_mlr_context(n_states):
    config = _best_mlr_config(n_states)
    states, exog = _read_states_and_exog(n_states)
    X, y, meta, state_classes = build_transition_dataset(
        states,
        exog,
        max_state_lag=int(config['STATE_LAG']),
        max_exog_lag=int(config['EXOG_LAG']),
    )
    return X, y, meta, state_classes, split_transition_dataset(X, y, meta), config


def _load_mlg_context(n_states):
    report = pd.read_csv(BASE_DIR / 'phase_1' / f'mlg_transition_report_q{n_states}.csv')
    config = best_mlg_config(report)
    states, exog = _read_states_and_exog(n_states)
    X, y, meta, state_classes = build_transition_dataset(
        states,
        exog,
        max_state_lag=int(config['STATE_LAG']),
        max_exog_lag=int(config['EXOG_LAG']),
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']
    transformer = MLGFeatureTransformer(n_knots=int(config['N_KNOTS'])).fit(X_train)
    selected_terms = str(config['SELECTED_TERMS']).split('|')
    unknown = set(selected_terms) - set(transformer.term_names_)
    if unknown:
        raise ValueError(f'Saved q{n_states} MLG terms are unavailable: {sorted(unknown)}')
    prepared = PreparedMLGDesign(
        transformer=transformer,
        selected_terms=selected_terms,
        term_p_values=json.loads(config['TERM_P_VALUES']),
        X_train=X_train,
        y_train=y_train,
        inference_method=str(config.get('INFERENCE_METHOD', 'saved_selection')),
    )
    model = MLGTransitionModel(prepared, float(config['REGULARIZATION_C']))
    return X, y, meta, state_classes, splits, config, model


@dataclass
class CandidatePredictor:
    model: object
    state_classes: list
    feature_columns: list = None
    transformer: object = None
    terms: list = None
    duration_mean: float = None
    duration_scale: float = None

    def predict_proba(self, X):
        if self.transformer is not None:
            design = self.transformer.transform(X, self.terms)
            if self.duration_mean is not None:
                duration = np.log1p(np.clip(X['DURATION'].to_numpy(dtype=float), 0, None))
                duration = ((duration - self.duration_mean) / self.duration_scale).reshape(-1, 1)
                design = np.hstack([design, duration])
            raw = self.model.predict_proba(design)
            classes = self.model.classes_
        else:
            frame = X[self.feature_columns].copy()
            if self.duration_mean is not None:
                frame['DURATION_LOG1P'] = np.log1p(
                    np.clip(X['DURATION'].to_numpy(dtype=float), 0, None)
                )
            raw = self.model.predict_proba(frame)
            classes = self.model.named_steps['logit'].classes_
        return _align_probabilities(raw, classes, self.state_classes)


def _fit_classifier(design, y, regularization_c=1.0):
    model = LogisticRegression(
        C=float(regularization_c),
        penalty='l2',
        solver='lbfgs',
        max_iter=4000,
        random_state=DEFAULT_RANDOM_SEED,
    )
    model.fit(design, y)
    return model


def fit_mlr_candidates(X_train, y_train, state_classes):
    groups = grouped_columns(X_train.columns)
    candidate_columns = {
        'state_only': groups['state'],
        'state_market': groups['state'] + groups['market'],
        'state_duration': groups['state'] + groups['duration'],
        'state_market_restricted_duration': groups['state'] + groups['market'],
        'full': list(X_train.columns),
    }
    candidates = {}
    for name, columns in candidate_columns.items():
        fit_frame = X_train[columns].copy()
        restricted = name == 'state_market_restricted_duration'
        if restricted:
            fit_frame['DURATION_LOG1P'] = np.log1p(
                np.clip(X_train['DURATION'].to_numpy(dtype=float), 0, None)
            )
        model = Pipeline([
            ('scale', StandardScaler()),
            ('logit', LogisticRegression(max_iter=4000, solver='lbfgs', random_state=42)),
        ])
        model.fit(fit_frame, y_train)
        candidates[name] = CandidatePredictor(
            model=model,
            state_classes=list(state_classes),
            feature_columns=columns,
            duration_mean=0.0 if restricted else None,
        )
    return candidates


def _mlg_selected_groups(model):
    groups = {'state': [], 'market': [], 'duration': []}
    for term in model.selected_terms:
        groups[feature_group(term)].append(term)
    return groups


def fit_mlg_candidates(model, X_train, y_train, state_classes):
    groups = _mlg_selected_groups(model)
    candidate_terms = {
        'state_only': groups['state'],
        'state_market': groups['state'] + groups['market'],
        'state_duration': groups['state'] + groups['duration'],
        'state_market_restricted_duration': groups['state'] + groups['market'],
        'full': list(model.selected_terms),
    }
    candidates = {}
    for name, terms in candidate_terms.items():
        design = model.transformer.transform(X_train, terms)
        restricted = name == 'state_market_restricted_duration'
        duration_mean = None
        duration_scale = None
        if restricted:
            duration = np.log1p(np.clip(X_train['DURATION'].to_numpy(dtype=float), 0, None))
            duration_mean = float(duration.mean())
            duration_scale = float(duration.std()) or 1.0
            design = np.hstack([design, ((duration - duration_mean) / duration_scale).reshape(-1, 1)])
        classifier = _fit_classifier(design, y_train, model.regularization_c)
        candidates[name] = CandidatePredictor(
            model=classifier,
            state_classes=list(state_classes),
            transformer=model.transformer,
            terms=terms,
            duration_mean=duration_mean,
            duration_scale=duration_scale,
        )
    return candidates


def evaluate_ablation_candidates(n_states, model_name, candidates, splits, state_classes):
    rows = []
    pair_rows = []
    for split_name in ['validation', 'prediction']:
        X_eval, y_eval, meta_eval = splits[split_name]
        current = meta_eval['CURRENT_STATE'].to_numpy(dtype=int)
        for candidate_name in CANDIDATE_ORDER:
            probabilities = candidates[candidate_name].predict_proba(X_eval)
            row = probability_metrics(y_eval, current, probabilities, state_classes)
            row.update({
                'N_STATES': n_states,
                'MODEL': model_name,
                'CANDIDATE': candidate_name,
                'SPLIT': split_name,
                'SPLIT_START': X_eval.index.min().date(),
                'SPLIT_END': X_eval.index.max().date(),
                'EVAL_ROWS': len(X_eval),
            })
            rows.append(row)

            class_to_column = {state: index for index, state in enumerate(state_classes)}
            y_values = y_eval.to_numpy(dtype=int)
            for from_state in state_classes:
                for to_state in state_classes:
                    if from_state == to_state:
                        continue
                    mask = (current == from_state) & (y_values == to_state)
                    if not mask.any():
                        continue
                    assigned = probabilities[mask, class_to_column[to_state]]
                    pair_rows.append({
                        'N_STATES': n_states,
                        'MODEL': model_name,
                        'CANDIDATE': candidate_name,
                        'SPLIT': split_name,
                        'FROM_STATE': from_state,
                        'TO_STATE': to_state,
                        'OBSERVED_TRANSITIONS': int(mask.sum()),
                        'MEAN_PROBABILITY_ASSIGNED_TO_DESTINATION': float(assigned.mean()),
                        'MEDIAN_PROBABILITY_ASSIGNED_TO_DESTINATION': float(np.median(assigned)),
                        'MEAN_DESTINATION_LOG_LOSS': float(-np.log(np.clip(assigned, 1e-12, 1.0)).mean()),
                    })
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def ablation_deltas(metrics):
    metric_columns = [
        'LOG_LOSS', 'BRIER_SCORE', 'ACCURACY', 'ECE',
        'EXIT_LOG_LOSS', 'EXIT_BRIER_SCORE', 'EXIT_ROC_AUC',
        'EXIT_AVERAGE_PRECISION', 'EXIT_RECALL_AT_0_5',
    ]
    rows = []
    for (model_name, split_name), group in metrics.groupby(['MODEL', 'SPLIT']):
        baseline = group[group['CANDIDATE'] == 'full'].iloc[0]
        for row in group.itertuples(index=False):
            record = {
                'N_STATES': row.N_STATES,
                'MODEL': model_name,
                'CANDIDATE': row.CANDIDATE,
                'SPLIT': split_name,
            }
            for metric in metric_columns:
                record[f'DELTA_{metric}_VERSUS_FULL'] = getattr(row, metric) - baseline[metric]
            rows.append(record)
    return pd.DataFrame(rows)


def _model_group_columns(model_name, X, mlg_model=None):
    if model_name == 'MLR':
        return grouped_columns(X.columns)
    selected_columns = []
    for term in mlg_model.selected_terms:
        selected_columns.extend(mlg_model.transformer.term_spec(term).columns)
    return grouped_columns(selected_columns)


def conditional_permutation_analysis(
    n_states,
    model_name,
    full_predictor,
    splits,
    state_classes,
    feature_groups,
    n_permutations=DEFAULT_PERMUTATIONS,
    block_length=DEFAULT_BLOCK_LENGTH,
    random_seed=DEFAULT_RANDOM_SEED,
):
    rng = np.random.default_rng(random_seed + n_states + (0 if model_name == 'MLR' else 1000))
    rows = []
    metric_names = ['LOG_LOSS', 'BRIER_SCORE', 'ACCURACY', 'ECE', 'EXIT_LOG_LOSS', 'EXIT_BRIER_SCORE']
    for split_name in ['validation', 'prediction']:
        X_eval, y_eval, meta_eval = splits[split_name]
        current = meta_eval['CURRENT_STATE'].to_numpy(dtype=int)
        baseline_probabilities = full_predictor.predict_proba(X_eval)
        baseline = probability_metrics(y_eval, current, baseline_probabilities, state_classes)
        for group_name in ['market', 'duration']:
            columns = feature_groups[group_name]
            if not columns:
                continue
            for replicate in range(int(n_permutations)):
                permuted = state_conditional_block_permutation(
                    X_eval,
                    columns,
                    current,
                    block_length,
                    rng,
                )
                probabilities = full_predictor.predict_proba(permuted)
                values = probability_metrics(y_eval, current, probabilities, state_classes)
                row = {
                    'N_STATES': n_states,
                    'MODEL': model_name,
                    'SPLIT': split_name,
                    'FEATURE_GROUP': group_name,
                    'REPLICATE': replicate + 1,
                    'BLOCK_LENGTH': int(block_length),
                    'GROUP_COLUMN_COUNT': len(columns),
                }
                for metric in metric_names:
                    row[f'BASELINE_{metric}'] = baseline[metric]
                    row[f'PERMUTED_{metric}'] = values[metric]
                    row[f'DELTA_{metric}'] = values[metric] - baseline[metric]
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_permutations(replicates):
    delta_columns = [column for column in replicates if column.startswith('DELTA_')]
    rows = []
    for keys, group in replicates.groupby(['N_STATES', 'MODEL', 'SPLIT', 'FEATURE_GROUP']):
        row = dict(zip(['N_STATES', 'MODEL', 'SPLIT', 'FEATURE_GROUP'], keys))
        row['PERMUTATIONS'] = len(group)
        row['BLOCK_LENGTH'] = int(group['BLOCK_LENGTH'].iloc[0])
        row['GROUP_COLUMN_COUNT'] = int(group['GROUP_COLUMN_COUNT'].iloc[0])
        for column in delta_columns:
            values = group[column].to_numpy(dtype=float)
            row[f'{column}_MEAN'] = float(np.mean(values))
            row[f'{column}_MEDIAN'] = float(np.median(values))
            row[f'{column}_CI_LOWER'] = float(np.quantile(values, 0.025))
            row[f'{column}_CI_UPPER'] = float(np.quantile(values, 0.975))
            row[f'{column}_POSITIVE_SHARE'] = float(np.mean(values > 0))
            row[f'{column}_ONE_SIDED_P_VALUE'] = float((np.sum(values <= 0) + 1) / (len(values) + 1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablation_deltas(deltas, path):
    candidates = [name for name in CANDIDATE_ORDER if name != 'full']
    labels = ['State only', 'State + market', 'State + duration', 'State + market +\nrestricted duration']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    colors = {'MLR': '#C25B3E', 'MLG': '#287D8E'}
    width = 0.36
    x = np.arange(len(candidates))
    for ax, split_name in zip(axes, ['validation', 'prediction']):
        split = deltas[deltas['SPLIT'] == split_name]
        for offset, model_name in zip([-width / 2, width / 2], ['MLR', 'MLG']):
            values = []
            for candidate in candidates:
                match = split[(split['MODEL'] == model_name) & (split['CANDIDATE'] == candidate)]
                values.append(float(match['DELTA_LOG_LOSS_VERSUS_FULL'].iloc[0]))
            ax.bar(x + offset, values, width=width, color=colors[model_name], label=model_name)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha='right')
        ax.set_title(split_name.title())
        ax.grid(axis='y', alpha=0.25)
    axes[0].set_ylabel('Change in log loss versus full model\n(positive means worse)')
    axes[1].legend(loc='best')
    fig.suptitle(f"q{int(deltas['N_STATES'].iloc[0])} feature-group ablation")
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_permutation_summary(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, constrained_layout=True)
    colors = {'MLR': '#C25B3E', 'MLG': '#287D8E'}
    groups = ['market', 'duration']
    labels = ['Market indices', 'Duration']
    width = 0.36
    x = np.arange(len(groups))
    for ax, split_name in zip(axes, ['validation', 'prediction']):
        split = summary[summary['SPLIT'] == split_name]
        for offset, model_name in zip([-width / 2, width / 2], ['MLR', 'MLG']):
            means = []
            lower = []
            upper = []
            for group_name in groups:
                row = split[(split['MODEL'] == model_name) & (split['FEATURE_GROUP'] == group_name)].iloc[0]
                mean = float(row['DELTA_LOG_LOSS_MEAN'])
                means.append(mean)
                lower.append(mean - float(row['DELTA_LOG_LOSS_CI_LOWER']))
                upper.append(float(row['DELTA_LOG_LOSS_CI_UPPER']) - mean)
            ax.bar(x + offset, means, width=width, color=colors[model_name], label=model_name)
            ax.errorbar(x + offset, means, yerr=[lower, upper], fmt='none', ecolor='black', capsize=3)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(split_name.title())
        ax.grid(axis='y', alpha=0.25)
    axes[0].set_ylabel('Mean change in log loss after permutation\n(positive means useful information was removed)')
    axes[1].legend(loc='best')
    fig.suptitle(f"q{int(summary['N_STATES'].iloc[0])} state-conditional block permutation")
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def run_feature_ablation_case(
    n_states,
    output_dir=OUTPUT_DIR,
    n_permutations=DEFAULT_PERMUTATIONS,
    block_length=DEFAULT_BLOCK_LENGTH,
    random_seed=DEFAULT_RANDOM_SEED,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, mlr_classes, mlr_splits, mlr_config = _load_mlr_context(n_states)
    _, _, _, mlg_classes, mlg_splits, mlg_config, selected_mlg = _load_mlg_context(n_states)
    if list(mlr_classes) != list(mlg_classes):
        raise ValueError(f'MLR and MLG state classes differ for q{n_states}.')

    mlr_X_train, mlr_y_train, _ = mlr_splits['train']
    mlg_X_train, mlg_y_train, _ = mlg_splits['train']
    mlr_candidates = fit_mlr_candidates(mlr_X_train, mlr_y_train, mlr_classes)
    mlg_candidates = fit_mlg_candidates(selected_mlg, mlg_X_train, mlg_y_train, mlg_classes)

    mlr_metrics, mlr_pairs = evaluate_ablation_candidates(
        n_states, 'MLR', mlr_candidates, mlr_splits, mlr_classes
    )
    mlg_metrics, mlg_pairs = evaluate_ablation_candidates(
        n_states, 'MLG', mlg_candidates, mlg_splits, mlg_classes
    )
    metrics = pd.concat([mlr_metrics, mlg_metrics], ignore_index=True)
    pairs = pd.concat([mlr_pairs, mlg_pairs], ignore_index=True)
    deltas = ablation_deltas(metrics)

    mlr_groups = _model_group_columns('MLR', mlr_X_train)
    mlg_groups = _model_group_columns('MLG', mlg_X_train, selected_mlg)
    mlr_permutations = conditional_permutation_analysis(
        n_states,
        'MLR',
        mlr_candidates['full'],
        mlr_splits,
        mlr_classes,
        mlr_groups,
        n_permutations,
        block_length,
        random_seed,
    )
    mlg_permutations = conditional_permutation_analysis(
        n_states,
        'MLG',
        mlg_candidates['full'],
        mlg_splits,
        mlg_classes,
        mlg_groups,
        n_permutations,
        block_length,
        random_seed,
    )
    permutations = pd.concat([mlr_permutations, mlg_permutations], ignore_index=True)
    permutation_summary = summarize_permutations(permutations)

    paths = {
        'metrics': output_dir / f'feature_ablation_metrics_q{n_states}.csv',
        'deltas': output_dir / f'feature_ablation_deltas_q{n_states}.csv',
        'pairs': output_dir / f'feature_ablation_pair_metrics_q{n_states}.csv',
        'permutations': output_dir / f'conditional_permutation_replicates_q{n_states}.csv',
        'permutation_summary': output_dir / f'conditional_permutation_summary_q{n_states}.csv',
        'ablation_plot': output_dir / f'feature_ablation_log_loss_q{n_states}.png',
        'permutation_plot': output_dir / f'conditional_permutation_log_loss_q{n_states}.png',
        'config': output_dir / f'feature_ablation_run_config_q{n_states}.csv',
    }
    metrics.to_csv(paths['metrics'], index=False)
    deltas.to_csv(paths['deltas'], index=False)
    pairs.to_csv(paths['pairs'], index=False)
    permutations.to_csv(paths['permutations'], index=False)
    permutation_summary.to_csv(paths['permutation_summary'], index=False)
    plot_ablation_deltas(deltas, paths['ablation_plot'])
    plot_permutation_summary(permutation_summary, paths['permutation_plot'])

    config = pd.DataFrame([{
        'N_STATES': n_states,
        'MLR_STATE_LAG': int(mlr_config['STATE_LAG']),
        'MLR_EXOG_LAG': int(mlr_config['EXOG_LAG']),
        'MLG_STATE_LAG': int(mlg_config['STATE_LAG']),
        'MLG_EXOG_LAG': int(mlg_config['EXOG_LAG']),
        'MLG_N_KNOTS': int(mlg_config['N_KNOTS']),
        'MLG_REGULARIZATION_C': float(mlg_config['REGULARIZATION_C']),
        'MLG_SELECTED_TERMS': str(mlg_config['SELECTED_TERMS']),
        'RESTRICTED_DURATION_SPECIFICATION': 'single_global_linear_log1p',
        'PERMUTATION_METHOD': 'joint fixed-block permutation within current state',
        'PERMUTATION_BLOCK_LENGTH': int(block_length),
        'PERMUTATION_REPLICATES': int(n_permutations),
        'RANDOM_SEED': int(random_seed),
        'PRODUCTION_MODEL_SELECTION_MODIFIED': False,
    }])
    config.to_csv(paths['config'], index=False)
    return {
        'metrics': metrics,
        'deltas': deltas,
        'pairs': pairs,
        'permutations': permutations,
        'permutation_summary': permutation_summary,
        'paths': paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, nargs='+', default=list(DEFAULT_STATES))
    parser.add_argument('--permutations', type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument('--block-length', type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    for n_states in args.states:
        run_feature_ablation_case(
            n_states,
            n_permutations=args.permutations,
            block_length=args.block_length,
            random_seed=args.random_seed,
        )
        print(f'Completed q{n_states} transition feature-ablation diagnostics.')
    print(f'Outputs saved under: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
