"""Group-weighted regularized multinomial logistic GAM transition model."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from p1_3_transition_features import build_transition_dataset
from p1_4_dynamic_transition import (
    _align_probabilities,
    evaluate_probabilities,
    split_transition_dataset,
)
from p1_4_mlg_transition import (
    MLGFeatureTransformer,
    PreparedMLGDesign,
    SIGNIFICANCE_LEVEL,
    _partial_effect_frame,
    load_or_create_mlg_report,
)
from project_config import BASE_DIR


MODEL_NAME = 'group_weighted_multinomial_logistic_gam'
DEFAULT_STATE_WEIGHTS = (1.0,)
DEFAULT_MARKET_WEIGHTS = (0.5, 1.0)
DEFAULT_DURATION_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
RANDOM_STATE = 42


def term_group(term_name):
    if term_name.startswith('STATE_LAG_'):
        return 'state'
    if term_name.startswith('DURATION'):
        return 'duration'
    return 'market'


def validate_group_weights(group_weights):
    expected = {'state', 'market', 'duration'}
    missing = expected - set(group_weights)
    if missing:
        raise ValueError(f'Missing group weights: {sorted(missing)}')
    weights = {name: float(group_weights[name]) for name in expected}
    if any(not np.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError('Every group weight must be finite and strictly positive.')
    return weights


class GroupWeightedMLGTransitionModel:
    """MLG fitted with a distinct L2 penalty multiplier for each term group."""

    def __init__(self, prepared_design, regularization_c, group_weights, random_state=RANDOM_STATE):
        self.transformer = prepared_design.transformer
        self.selected_terms = list(prepared_design.selected_terms)
        self.term_p_values = dict(prepared_design.term_p_values)
        self.inference_method = prepared_design.inference_method
        self.regularization_c = float(regularization_c)
        self.group_weights = validate_group_weights(group_weights)
        self.term_penalty_weights = {
            term: self.group_weights[term_group(term)] for term in self.selected_terms
        }
        design = self.transformer.transform(prepared_design.X_train, self.selected_terms)
        self.column_multipliers_ = self._column_multipliers(prepared_design.X_train)
        weighted_design = design * self.column_multipliers_
        self.classifier = LogisticRegression(
            C=self.regularization_c,
            penalty='l2',
            solver='lbfgs',
            max_iter=4000,
            random_state=random_state,
        )
        self.classifier.fit(weighted_design, prepared_design.y_train)

    @property
    def classes_(self):
        return self.classifier.classes_

    @property
    def effective_coefficients_(self):
        """Coefficients on the original penalized MLG design, before group scaling."""
        return self.classifier.coef_ * self.column_multipliers_

    def _column_multipliers(self, X):
        slices = self.transformer.term_slices(X, self.selected_terms)
        width = max(term_slice.stop for term_slice in slices.values())
        multipliers = np.ones(width, dtype=float)
        for term, term_slice in slices.items():
            multipliers[term_slice] = 1.0 / np.sqrt(self.term_penalty_weights[term])
        return multipliers

    def predict_proba(self, X):
        design = self.transformer.transform(X, self.selected_terms)
        return self.classifier.predict_proba(design * self.column_multipliers_)

    def term_log_odds(self, X, term_name):
        if term_name not in self.selected_terms:
            raise ValueError(f'{term_name} is not in the selected weighted MLG model.')

        term_matrix = self.transformer.transform_term(X, term_name)
        slices = self.transformer.term_slices(X, self.selected_terms)
        coefficients = self.effective_coefficients_[:, slices[term_name]]

        if len(self.classes_) == 2:
            effects = term_matrix @ coefficients[0]
            return effects.reshape(-1, 1), [f'{self.classes_[1]} vs {self.classes_[0]}']

        duration_match = re.match(r'^DURATION_STATE_(\d+)$', term_name)
        if duration_match:
            focal_state = int(duration_match.group(1))
            if focal_state in set(self.classes_):
                focal_index = int(np.where(self.classes_ == focal_state)[0][0])
                effects = []
                labels = []
                for other_index, other_state in enumerate(self.classes_):
                    if other_index == focal_index:
                        continue
                    effects.append(
                        term_matrix @ (coefficients[focal_index] - coefficients[other_index])
                    )
                    labels.append(f'{focal_state} vs {other_state}')
                return np.column_stack(effects), labels

        effects = []
        labels = []
        for reference_index in range(len(self.classes_) - 1):
            for contrast_index in range(reference_index + 1, len(self.classes_)):
                effects.append(
                    term_matrix @ (coefficients[contrast_index] - coefficients[reference_index])
                )
                labels.append(
                    f'{self.classes_[contrast_index]} vs {self.classes_[reference_index]}'
                )
        return np.column_stack(effects), labels


def _read_transition_inputs(n_states):
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


def _prepared_from_source_row(X_train, y_train, source_row):
    transformer = MLGFeatureTransformer(n_knots=int(source_row['N_KNOTS'])).fit(X_train)
    selected_terms = str(source_row['SELECTED_TERMS']).split('|')
    unknown = set(selected_terms) - set(transformer.term_names_)
    if unknown:
        raise ValueError(f'Saved MLG terms are unavailable: {sorted(unknown)}')
    return PreparedMLGDesign(
        transformer=transformer,
        selected_terms=selected_terms,
        term_p_values=json.loads(source_row['TERM_P_VALUES']),
        X_train=X_train,
        y_train=y_train,
        inference_method=str(source_row.get('INFERENCE_METHOD', 'saved_mlg_screen')),
    )


def _source_architectures(original_report):
    source = original_report[original_report['SPLIT'] == 'validation'].copy()
    if source.empty:
        raise ValueError('The original MLG report has no validation candidates.')
    return source.sort_values(
        ['STATE_LAG', 'EXOG_LAG', 'N_KNOTS', 'REGULARIZATION_C']
    )


def run_weighted_mlg_grid(
    n_states,
    state_weights=DEFAULT_STATE_WEIGHTS,
    market_weights=DEFAULT_MARKET_WEIGHTS,
    duration_weights=DEFAULT_DURATION_WEIGHTS,
    original_report=None,
):
    """Refit every saved significant MLG architecture under group-weighted L2 loss."""
    original_report = (
        load_or_create_mlg_report(n_states) if original_report is None else original_report
    )
    source = _source_architectures(original_report)
    states, exog = _read_transition_inputs(n_states)
    rows = []
    errors = []

    architecture_columns = ['STATE_LAG', 'EXOG_LAG', 'N_KNOTS']
    for architecture, architecture_rows in source.groupby(architecture_columns, sort=True):
        state_lag, exog_lag, n_knots = (int(value) for value in architecture)
        source_row = architecture_rows.iloc[0]
        X, y, meta, state_classes = build_transition_dataset(
            states,
            exog,
            max_state_lag=state_lag,
            max_exog_lag=exog_lag,
        )
        splits = split_transition_dataset(X, y, meta)
        X_train, y_train, _ = splits['train']
        try:
            prepared = _prepared_from_source_row(X_train, y_train, source_row)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            errors.append({
                'N_STATES': n_states,
                'STATE_LAG': state_lag,
                'EXOG_LAG': exog_lag,
                'N_KNOTS': n_knots,
                'ERROR': str(error),
            })
            continue

        selected_p_values = {
            term: prepared.term_p_values[term] for term in prepared.selected_terms
        }
        regularization_values = sorted(
            architecture_rows['REGULARIZATION_C'].astype(float).unique()
        )
        for regularization_c in regularization_values:
            for state_weight in state_weights:
                for market_weight in market_weights:
                    for duration_weight in duration_weights:
                        weights = {
                            'state': state_weight,
                            'market': market_weight,
                            'duration': duration_weight,
                        }
                        try:
                            model = GroupWeightedMLGTransitionModel(
                                prepared,
                                regularization_c,
                                weights,
                            )
                        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                            errors.append({
                                'N_STATES': n_states,
                                'STATE_LAG': state_lag,
                                'EXOG_LAG': exog_lag,
                                'N_KNOTS': n_knots,
                                'REGULARIZATION_C': regularization_c,
                                'STATE_PENALTY_WEIGHT': state_weight,
                                'MARKET_PENALTY_WEIGHT': market_weight,
                                'DURATION_PENALTY_WEIGHT': duration_weight,
                                'ERROR': str(error),
                            })
                            continue

                        common = {
                            'N_STATES': n_states,
                            'STATE_LAG': state_lag,
                            'EXOG_LAG': exog_lag,
                            'N_KNOTS': n_knots,
                            'SIGNIFICANCE_LEVEL': float(source_row['SIGNIFICANCE_LEVEL']),
                            'SELECTED_TERM_COUNT': len(prepared.selected_terms),
                            'SELECTED_TERMS': '|'.join(prepared.selected_terms),
                            'MAX_SELECTED_P_VALUE': max(selected_p_values.values()),
                            'ALL_TERMS_SIGNIFICANT': all(
                                value <= SIGNIFICANCE_LEVEL for value in selected_p_values.values()
                            ),
                            'TERM_P_VALUES': json.dumps(prepared.term_p_values, sort_keys=True),
                            'INFERENCE_METHOD': prepared.inference_method,
                            'SOURCE_MODEL': 'multinomial_logistic_gam',
                            'LOSS_FUNCTION': 'multinomial_nll_plus_group_weighted_l2',
                            'REGULARIZATION_C': float(regularization_c),
                            'SMOOTHING_LAMBDA': 1.0 / float(regularization_c),
                            'STATE_PENALTY_WEIGHT': float(state_weight),
                            'MARKET_PENALTY_WEIGHT': float(market_weight),
                            'DURATION_PENALTY_WEIGHT': float(duration_weight),
                            'MARKET_TO_STATE_PENALTY_RATIO': float(market_weight / state_weight),
                            'DURATION_TO_STATE_PENALTY_RATIO': float(duration_weight / state_weight),
                            'DURATION_TO_MARKET_PENALTY_RATIO': float(duration_weight / market_weight),
                            'TRAIN_ROWS': len(X_train),
                        }
                        for split_name in ['validation', 'prediction']:
                            X_eval, y_eval, _ = splits[split_name]
                            probabilities = _align_probabilities(
                                model.predict_proba(X_eval),
                                model.classes_,
                                state_classes,
                            )
                            extra = dict(common)
                            extra.update({
                                'SPLIT': split_name,
                                'SPLIT_START': X_eval.index.min().date(),
                                'SPLIT_END': X_eval.index.max().date(),
                                'EVAL_ROWS': len(X_eval),
                            })
                            rows.append(evaluate_probabilities(
                                MODEL_NAME,
                                y_eval,
                                probabilities,
                                state_classes,
                                extra,
                            ))

    error_path = BASE_DIR / 'phase_1' / f'weighted_mlg_transition_errors_q{n_states}.csv'
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()
    if not rows:
        raise RuntimeError(f'Every group-weighted MLG candidate failed for q{n_states}.')
    return pd.DataFrame(rows)


def best_weighted_mlg_config(report):
    significant = report['ALL_TERMS_SIGNIFICANT']
    if significant.dtype != bool:
        significant = significant.astype(str).str.lower().eq('true')
    candidates = report[
        (report['MODEL'] == MODEL_NAME)
        & (report['SPLIT'] == 'validation')
        & significant
    ].copy()
    if candidates.empty:
        raise ValueError('No significant weighted MLG validation candidate is available.')
    return candidates.sort_values(
        [
            'LOG_LOSS', 'BRIER_SCORE', 'ACCURACY', 'SELECTED_TERM_COUNT',
            'STATE_LAG', 'EXOG_LAG', 'N_KNOTS', 'REGULARIZATION_C',
            'DURATION_PENALTY_WEIGHT', 'MARKET_PENALTY_WEIGHT',
        ],
        ascending=[True, True, False, True, True, True, True, True, False, True],
    ).iloc[0]


def save_weighted_mlg_transition_report(n_states, **grid_options):
    report = run_weighted_mlg_grid(n_states, **grid_options)
    report_path = BASE_DIR / 'phase_1' / f'weighted_mlg_transition_report_q{n_states}.csv'
    report.to_csv(report_path, index=False)
    best = best_weighted_mlg_config(report)
    pd.DataFrame([best]).to_csv(
        BASE_DIR / 'phase_1' / f'weighted_mlg_selected_config_q{n_states}.csv',
        index=False,
    )

    p_values = json.loads(best['TERM_P_VALUES'])
    selected_terms = set(str(best['SELECTED_TERMS']).split('|'))
    significance = pd.DataFrame([
        {
            'N_STATES': n_states,
            'TERM': term,
            'P_VALUE': p_value,
            'SELECTED': term in selected_terms,
            'SIGNIFICANT_AT_5_PERCENT': p_value <= SIGNIFICANCE_LEVEL,
            'PENALTY_GROUP': term_group(term),
            'SELECTED_GROUP_WEIGHT': float(best[f'{term_group(term).upper()}_PENALTY_WEIGHT']),
            'INFERENCE_SOURCE': 'original_mlg_significance_screen',
        }
        for term, p_value in p_values.items()
    ])
    significance.to_csv(
        BASE_DIR / 'phase_1' / f'weighted_mlg_selected_significance_q{n_states}.csv',
        index=False,
    )
    return report, report_path


def load_or_create_weighted_mlg_report(n_states):
    path = BASE_DIR / 'phase_1' / f'weighted_mlg_transition_report_q{n_states}.csv'
    required = {
        'STATE_PENALTY_WEIGHT', 'MARKET_PENALTY_WEIGHT',
        'DURATION_PENALTY_WEIGHT', 'TERM_P_VALUES',
    }
    if path.exists():
        report = pd.read_csv(path)
        if required.issubset(report.columns):
            return report
    report, _ = save_weighted_mlg_transition_report(n_states)
    return report


def fit_selected_weighted_mlg_transition(n_states, report=None):
    report = (
        load_or_create_weighted_mlg_report(n_states) if report is None else report
    )
    best = best_weighted_mlg_config(report)
    states, exog = _read_transition_inputs(n_states)
    X, y, meta, state_classes = build_transition_dataset(
        states,
        exog,
        max_state_lag=int(best['STATE_LAG']),
        max_exog_lag=int(best['EXOG_LAG']),
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']
    prepared = _prepared_from_source_row(X_train, y_train, best)
    model = GroupWeightedMLGTransitionModel(
        prepared,
        float(best['REGULARIZATION_C']),
        {
            'state': float(best['STATE_PENALTY_WEIGHT']),
            'market': float(best['MARKET_PENALTY_WEIGHT']),
            'duration': float(best['DURATION_PENALTY_WEIGHT']),
        },
    )
    return model, splits, state_classes, best


def plot_selected_weighted_mlg_partial_effects(n_states, model, X_train, output_dir):
    output_dir = Path(output_dir) / f'weighted_mlg_partial_effects_q{n_states}'
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []

    for term_name in model.selected_terms:
        values, labels, effects, contrasts, kind = _partial_effect_frame(
            model, X_train, term_name
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for column, contrast in enumerate(contrasts):
            if kind == 'categorical':
                ax.plot(values, effects[:, column], marker='o', label=contrast)
            else:
                ax.plot(values, effects[:, column], linewidth=2, label=contrast)
            for row, value in enumerate(values):
                records.append({
                    'N_STATES': n_states,
                    'MODEL': MODEL_NAME,
                    'TERM': term_name,
                    'P_VALUE': model.term_p_values[term_name],
                    'PENALTY_GROUP': term_group(term_name),
                    'GROUP_PENALTY_WEIGHT': model.term_penalty_weights[term_name],
                    'PREDICTOR_VALUE': value,
                    'PREDICTOR_LEVEL': labels[row] if kind == 'categorical' else None,
                    'LOG_ODDS_CONTRAST': contrast,
                    'CENTERED_PARTIAL_EFFECT': effects[row, column],
                })

        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_title(f'q{n_states} weighted MLG partial effect: {term_name}')
        ax.set_xlabel(term_name)
        ax.set_ylabel('Centered contribution to log-odds')
        if kind == 'categorical':
            ax.set_xticks(values)
            ax.set_xticklabels(labels)
        ax.legend(title='Next-state contrast', loc='best')
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', term_name)
        path = output_dir / f'{safe_name}.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)

    effects_path = output_dir / 'partial_effects.csv'
    pd.DataFrame(records).to_csv(effects_path, index=False)
    return paths, effects_path


def main():
    for n_states in (2, 4):
        _, path = save_weighted_mlg_transition_report(n_states)
        print(f'Weighted MLG q{n_states} report saved to: {path}')


if __name__ == '__main__':
    main()
