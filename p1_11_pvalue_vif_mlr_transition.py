"""Multinomial logistic transition model selected with p-values and VIF."""

from dataclasses import dataclass
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from p1_3_transition_features import compute_regime_duration
from p1_4_dynamic_transition import (
    _align_probabilities,
    evaluate_probabilities,
    split_transition_dataset,
)
from p1_4_mlg_transition import _fit_mnlogit_for_inference, _joint_wald_p_values
from project_config import BASE_DIR


MODEL_NAME = 'pvalue_vif_selected_mlr'
SIGNIFICANCE_LEVEL = 0.05
VIF_THRESHOLD = 5.0
REFERENCE_STATE = 0
RANDOM_STATE = 42


def predictor_group(term_name):
    if term_name.startswith('STATE_LAG_'):
        return 'state'
    if term_name.startswith('DURATION_STATE_'):
        return 'duration'
    return 'macro'


def build_corrected_transition_dataset(
    states,
    exog_rets,
    max_state_lag=1,
    max_exog_lag=1,
    reference_state=REFERENCE_STATE,
):
    """Build a full-rank MLR design with valid state lags and no global duration."""
    states = states.astype(int).rename('STATE')
    aligned = pd.concat([states, exog_rets], axis=1, join='inner').dropna()
    states = aligned['STATE'].astype(int)
    exog_rets = aligned.drop(columns=['STATE'])
    state_classes = sorted(states.unique())
    if reference_state not in state_classes:
        raise ValueError(f'Reference state {reference_state} is not available.')

    frames = []
    term_columns = {}
    for lag in range(int(max_state_lag) + 1):
        lagged = states.shift(lag)
        columns = []
        frame = pd.DataFrame(index=states.index)
        for state in state_classes:
            if state == reference_state:
                continue
            column = f'STATE_LAG_{lag}_{state}'
            frame[column] = (lagged == state).astype(float)
            columns.append(column)
        frame.loc[lagged.isna(), columns] = np.nan
        frames.append(frame)
        term_columns[f'STATE_LAG_{lag}'] = columns

    for lag in range(int(max_exog_lag) + 1):
        frame = exog_rets.shift(lag).copy()
        frame.columns = [f'{column}_LAG_{lag}' for column in frame.columns]
        frames.append(frame)
        for column in frame.columns:
            term_columns[column] = [column]

    duration = compute_regime_duration(states)['DURATION']
    duration_frame = pd.DataFrame(index=states.index)
    for state in state_classes:
        column = f'DURATION_STATE_{state}'
        duration_frame[column] = duration * (states == state).astype(int)
        term_columns[column] = [column]
    frames.append(duration_frame)

    X = pd.concat(frames, axis=1)
    y = states.shift(-1).rename('NEXT_STATE')
    meta = pd.DataFrame({'CURRENT_STATE': states}, index=states.index)
    dataset = pd.concat([X, y, meta], axis=1).dropna()
    X = dataset[X.columns].astype(float)
    if any(X[column].nunique() <= 1 for column in X.columns):
        constants = [column for column in X if X[column].nunique() <= 1]
        raise ValueError(f'Corrected MLR design contains constant columns: {constants}')

    return (
        X,
        dataset['NEXT_STATE'].astype(int),
        dataset[['CURRENT_STATE']].astype(int),
        state_classes,
        term_columns,
    )


def _active_columns(active_terms, term_columns):
    return [column for term in active_terms for column in term_columns[term]]


def _term_slices(active_terms, term_columns):
    slices = {}
    start = 0
    for term in active_terms:
        stop = start + len(term_columns[term])
        slices[term] = slice(start, stop)
        start = stop
    return slices


def feature_vif_values(design, feature_names):
    """Compute VIF with an intercept, retaining infinity for exact dependence."""
    matrix = np.asarray(design, dtype=float)
    with_intercept = np.column_stack([np.ones(len(matrix)), matrix])
    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for index, feature in enumerate(feature_names, start=1):
            try:
                value = float(variance_inflation_factor(with_intercept, index))
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                value = np.inf
            values[feature] = value
    return values


def _term_vif_values(active_terms, term_columns, feature_vifs):
    return {
        term: max(feature_vifs[column] for column in term_columns[term])
        for term in active_terms
    }


def _fit_selection_iteration(X_train, y_train, active_terms, term_columns):
    columns = _active_columns(active_terms, term_columns)
    scaler = StandardScaler().fit(X_train[columns])
    design = scaler.transform(X_train[columns])
    result = _fit_mnlogit_for_inference(design, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p_values = _joint_wald_p_values(result, _term_slices(active_terms, term_columns))
    p_values = {
        term: value if np.isfinite(value) else 1.0
        for term, value in p_values.items()
    }
    feature_vifs = feature_vif_values(design, columns)
    term_vifs = _term_vif_values(active_terms, term_columns, feature_vifs)
    return p_values, feature_vifs, term_vifs


def macro_vif_removal_allowed(term_name, p_value, alpha=SIGNIFICANCE_LEVEL):
    """A macro may be removed for VIF only when it is not significant."""
    return predictor_group(term_name) != 'macro' or p_value > alpha


def _vif_removal_priority(term_name):
    state_match = re.match(r'^STATE_LAG_(\d+)$', term_name)
    if state_match:
        lag = int(state_match.group(1))
        return 4 + lag if lag > 0 else 0
    if predictor_group(term_name) == 'duration':
        return 3
    return 2


@dataclass
class SelectionResult:
    selected_terms: list
    selected_columns: list
    term_p_values: dict
    term_vifs: dict
    feature_vifs: dict
    history: list
    scaler: StandardScaler


def select_terms_pvalue_vif(
    X_train,
    y_train,
    term_columns,
    alpha=SIGNIFICANCE_LEVEL,
    vif_threshold=VIF_THRESHOLD,
):
    """Backward-select terms until all p-values and VIFs meet their thresholds."""
    active_terms = list(term_columns)
    history = []
    iteration = 0

    while active_terms:
        iteration += 1
        p_values, feature_vifs, term_vifs = _fit_selection_iteration(
            X_train, y_train, active_terms, term_columns
        )
        high_vif = [term for term in active_terms if term_vifs[term] > vif_threshold]
        eligible_vif = [
            term for term in high_vif
            if macro_vif_removal_allowed(term, p_values[term], alpha)
        ]
        if eligible_vif:
            removed = max(
                eligible_vif,
                key=lambda term: (
                    _vif_removal_priority(term),
                    term_vifs[term],
                    p_values[term],
                    term,
                ),
            )
            history.append({
                'ITERATION': iteration,
                'REMOVED_TERM': removed,
                'REMOVAL_REASON': 'vif',
                'P_VALUE_AT_REMOVAL': p_values[removed],
                'TERM_MAX_VIF_AT_REMOVAL': term_vifs[removed],
            })
            active_terms.remove(removed)
            continue

        nonsignificant = [term for term in active_terms if p_values[term] > alpha]
        if nonsignificant:
            removed = max(nonsignificant, key=lambda term: (p_values[term], term))
            history.append({
                'ITERATION': iteration,
                'REMOVED_TERM': removed,
                'REMOVAL_REASON': 'p_value',
                'P_VALUE_AT_REMOVAL': p_values[removed],
                'TERM_MAX_VIF_AT_REMOVAL': term_vifs[removed],
            })
            active_terms.remove(removed)
            continue

        if high_vif:
            blocked = '|'.join(sorted(high_vif))
            raise RuntimeError(
                f'VIF remains above {vif_threshold:g} only for protected significant macros: {blocked}'
            )

        columns = _active_columns(active_terms, term_columns)
        scaler = StandardScaler().fit(X_train[columns])
        return SelectionResult(
            selected_terms=list(active_terms),
            selected_columns=columns,
            term_p_values=p_values,
            term_vifs=term_vifs,
            feature_vifs=feature_vifs,
            history=history,
            scaler=scaler,
        )

    raise RuntimeError('P-value/VIF selection removed every MLR term.')


class PValueVIFMLRModel:
    def __init__(self, selection, X_train, y_train):
        self.selection = selection
        self.selected_terms = list(selection.selected_terms)
        self.selected_columns = list(selection.selected_columns)
        self.term_p_values = dict(selection.term_p_values)
        self.term_vifs = dict(selection.term_vifs)
        self.feature_vifs = dict(selection.feature_vifs)
        self.pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('logit', LogisticRegression(
                max_iter=4000,
                solver='lbfgs',
                random_state=RANDOM_STATE,
            )),
        ])
        self.pipeline.fit(X_train[self.selected_columns], y_train)

    @property
    def classes_(self):
        return self.pipeline.named_steps['logit'].classes_

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X[self.selected_columns])


def _read_inputs(n_states):
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


def run_pvalue_vif_mlr_grid(
    n_states,
    state_lags=(0, 1, 2),
    exog_lags=(0, 1, 2),
    alpha=SIGNIFICANCE_LEVEL,
    vif_threshold=VIF_THRESHOLD,
):
    states, exog = _read_inputs(n_states)
    rows = []
    audit_rows = []
    errors = []

    for state_lag in state_lags:
        for exog_lag in exog_lags:
            try:
                X, y, meta, state_classes, term_columns = build_corrected_transition_dataset(
                    states,
                    exog,
                    max_state_lag=state_lag,
                    max_exog_lag=exog_lag,
                )
                splits = split_transition_dataset(X, y, meta)
                X_train, y_train, _ = splits['train']
                selection = select_terms_pvalue_vif(
                    X_train,
                    y_train,
                    term_columns,
                    alpha=alpha,
                    vif_threshold=vif_threshold,
                )
                model = PValueVIFMLRModel(selection, X_train, y_train)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                errors.append({
                    'N_STATES': n_states,
                    'STATE_LAG': state_lag,
                    'EXOG_LAG': exog_lag,
                    'ERROR': str(error),
                })
                continue

            for history_row in selection.history:
                audit = dict(history_row)
                audit.update({
                    'N_STATES': n_states,
                    'STATE_LAG': state_lag,
                    'EXOG_LAG': exog_lag,
                })
                audit_rows.append(audit)

            common = {
                'N_STATES': n_states,
                'STATE_LAG': state_lag,
                'EXOG_LAG': exog_lag,
                'REFERENCE_STATE': REFERENCE_STATE,
                'SIGNIFICANCE_LEVEL': float(alpha),
                'VIF_THRESHOLD': float(vif_threshold),
                'SELECTED_TERM_COUNT': len(selection.selected_terms),
                'SELECTED_FEATURE_COUNT': len(selection.selected_columns),
                'SELECTED_TERMS': '|'.join(selection.selected_terms),
                'SELECTED_FEATURES': '|'.join(selection.selected_columns),
                'TERM_P_VALUES': json.dumps(selection.term_p_values, sort_keys=True),
                'TERM_VIFS': json.dumps(selection.term_vifs, sort_keys=True),
                'FEATURE_VIFS': json.dumps(selection.feature_vifs, sort_keys=True),
                'MAX_SELECTED_P_VALUE': max(selection.term_p_values.values()),
                'MAX_SELECTED_VIF': max(selection.feature_vifs.values()),
                'ALL_TERMS_SIGNIFICANT': all(
                    value <= alpha for value in selection.term_p_values.values()
                ),
                'ALL_FEATURES_WITHIN_VIF_THRESHOLD': all(
                    value <= vif_threshold for value in selection.feature_vifs.values()
                ),
                'REMOVED_TERMS': '|'.join(row['REMOVED_TERM'] for row in selection.history),
                'GLOBAL_DURATION_REMOVED_AS_REDUNDANT': True,
                'STATE_LAG_DEFECT_CORRECTED': True,
                'TRAIN_ROWS': len(X_train),
            }
            for split_name in ['validation', 'prediction']:
                X_eval, y_eval, _ = splits[split_name]
                probabilities = _align_probabilities(
                    model.predict_proba(X_eval), model.classes_, state_classes
                )
                extra = dict(common)
                extra.update({
                    'SPLIT': split_name,
                    'SPLIT_START': X_eval.index.min().date(),
                    'SPLIT_END': X_eval.index.max().date(),
                    'EVAL_ROWS': len(X_eval),
                })
                rows.append(evaluate_probabilities(
                    MODEL_NAME, y_eval, probabilities, state_classes, extra
                ))

    error_path = BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_errors_q{n_states}.csv'
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    elif error_path.exists():
        error_path.unlink()
    if not rows:
        raise RuntimeError(f'Every p-value/VIF MLR candidate failed for q{n_states}.')
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def best_pvalue_vif_mlr_config(report):
    significant = report['ALL_TERMS_SIGNIFICANT']
    within_vif = report['ALL_FEATURES_WITHIN_VIF_THRESHOLD']
    if significant.dtype != bool:
        significant = significant.astype(str).str.lower().eq('true')
    if within_vif.dtype != bool:
        within_vif = within_vif.astype(str).str.lower().eq('true')
    candidates = report[
        (report['MODEL'] == MODEL_NAME)
        & (report['SPLIT'] == 'validation')
        & significant
        & within_vif
    ].copy()
    if candidates.empty:
        raise ValueError('No p-value/VIF-compliant MLR validation candidate is available.')
    return candidates.sort_values(
        ['LOG_LOSS', 'BRIER_SCORE', 'STATE_LAG', 'EXOG_LAG'],
        ascending=[True, True, True, True],
    ).iloc[0]


def save_pvalue_vif_mlr_report(n_states, **grid_options):
    report, audit = run_pvalue_vif_mlr_grid(n_states, **grid_options)
    report_path = BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_transition_report_q{n_states}.csv'
    audit_path = BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_selection_audit_q{n_states}.csv'
    report.to_csv(report_path, index=False)
    audit.to_csv(audit_path, index=False)

    best = best_pvalue_vif_mlr_config(report)
    pd.DataFrame([best]).to_csv(
        BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_selected_config_q{n_states}.csv',
        index=False,
    )
    p_values = json.loads(best['TERM_P_VALUES'])
    term_vifs = json.loads(best['TERM_VIFS'])
    pd.DataFrame([
        {
            'N_STATES': n_states,
            'TERM': term,
            'PREDICTOR_GROUP': predictor_group(term),
            'P_VALUE': p_value,
            'SIGNIFICANT_AT_5_PERCENT': p_value <= SIGNIFICANCE_LEVEL,
            'TERM_MAX_VIF': term_vifs[term],
            'WITHIN_VIF_THRESHOLD': term_vifs[term] <= VIF_THRESHOLD,
        }
        for term, p_value in p_values.items()
    ]).to_csv(
        BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_selected_significance_q{n_states}.csv',
        index=False,
    )
    feature_vifs = json.loads(best['FEATURE_VIFS'])
    pd.DataFrame([
        {
            'N_STATES': n_states,
            'FEATURE': feature,
            'VIF': vif,
            'VIF_THRESHOLD': VIF_THRESHOLD,
            'WITHIN_THRESHOLD': vif <= VIF_THRESHOLD,
        }
        for feature, vif in feature_vifs.items()
    ]).to_csv(
        BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_selected_vif_q{n_states}.csv',
        index=False,
    )
    return report, report_path


def load_or_create_pvalue_vif_mlr_report(n_states):
    path = BASE_DIR / 'phase_1' / f'pvalue_vif_mlr_transition_report_q{n_states}.csv'
    required = {'TERM_P_VALUES', 'FEATURE_VIFS', 'MAX_SELECTED_VIF'}
    if path.exists():
        report = pd.read_csv(path)
        if required.issubset(report.columns):
            return report
    report, _ = save_pvalue_vif_mlr_report(n_states)
    return report


def fit_selected_pvalue_vif_mlr(n_states, report=None):
    report = (
        load_or_create_pvalue_vif_mlr_report(n_states) if report is None else report
    )
    best = best_pvalue_vif_mlr_config(report)
    states, exog = _read_inputs(n_states)
    X, y, meta, state_classes, term_columns = build_corrected_transition_dataset(
        states,
        exog,
        max_state_lag=int(best['STATE_LAG']),
        max_exog_lag=int(best['EXOG_LAG']),
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']
    selected_terms = str(best['SELECTED_TERMS']).split('|')
    selected_columns = str(best['SELECTED_FEATURES']).split('|')
    selection = SelectionResult(
        selected_terms=selected_terms,
        selected_columns=selected_columns,
        term_p_values=json.loads(best['TERM_P_VALUES']),
        term_vifs=json.loads(best['TERM_VIFS']),
        feature_vifs=json.loads(best['FEATURE_VIFS']),
        history=[],
        scaler=StandardScaler().fit(X_train[selected_columns]),
    )
    model = PValueVIFMLRModel(selection, X_train, y_train)
    return model, splits, state_classes, best


def main():
    for n_states in (2, 4):
        _, path = save_pvalue_vif_mlr_report(n_states)
        print(f'P-value/VIF MLR q{n_states} report saved to: {path}')


if __name__ == '__main__':
    main()
