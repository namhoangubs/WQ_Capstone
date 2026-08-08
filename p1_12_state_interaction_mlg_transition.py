"""State-interaction multinomial logistic GAM for HMM regime transitions."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    BOOTSTRAP_RANDOM_STATE,
    DEFAULT_BOOTSTRAP_C,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_KNOTS,
    DEFAULT_REGULARIZATION_C,
    MLGFeatureTransformer,
    MLGTransitionModel,
    SIGNIFICANCE_LEVEL,
    _binned_empirical_log_odds,
    _bootstrap_term_p_value,
    _fit_mnlogit_for_inference,
    _joint_wald_p_values,
    _read_transition_inputs,
    prepare_mlg_design,
)
from project_config import BASE_DIR


MODEL_NAME = 'state_interaction_multinomial_logistic_gam'
MODEL_LABEL = 'State-Interaction MLG'
BASE_MODEL_NAME = 'duration_free_multinomial_logistic_gam'
MODEL_SPECIFICATION = 'current_state_and_macro_only_no_duration_v2'
INTERACTION_PREFIX = 'STATE_X_'
DEFAULT_MIN_ORIGIN_OBSERVATIONS = 50
DEFAULT_MIN_ORIGIN_EXITS = 10
DEFAULT_MIN_UNIQUE_VALUES = 20


def is_market_term(term_name):
    """Return whether a selected additive term is a lagged market variable."""
    return (
        re.fullmatch(r'[A-Za-z0-9]+_LAG_\d+', str(term_name)) is not None
        and not str(term_name).startswith('STATE_LAG_')
    )


def duration_free_feature_frame(X):
    """Keep current-state indicators and market variables, excluding duration."""
    keep = [
        column for column in X.columns
        if not str(column).startswith('DURATION')
    ]
    return X.loc[:, keep]


def interaction_name(main_term):
    return f'{INTERACTION_PREFIX}{main_term}'


def interaction_main_term(term_name):
    if not str(term_name).startswith(INTERACTION_PREFIX):
        return None
    return str(term_name)[len(INTERACTION_PREFIX):]


def _pipe_list(value):
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return []
    return [item for item in text.split('|') if item]


def _json_dict(value):
    if value is None or (not isinstance(value, dict) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return {}
    return json.loads(text)


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values."""
    names = list(p_values)
    if not names:
        return {}
    values = np.asarray([
        np.nan_to_num(float(p_values[name]), nan=1.0, posinf=1.0, neginf=1.0)
        for name in names
    ])
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.empty_like(ranked)
    running = 1.0
    count = len(ranked)
    for index in range(count - 1, -1, -1):
        running = min(running, ranked[index] * count / (index + 1))
        adjusted_ranked[index] = running
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return {name: float(adjusted[index]) for index, name in enumerate(names)}


def _state_columns(base_transformer):
    state_term = base_transformer.term_spec('STATE_LAG_0')
    return list(state_term.columns)


def _state_from_column(column):
    return int(str(column).rsplit('_', 1)[-1])


def interaction_support_table(
    X_train,
    y_train,
    base_transformer,
    market_terms,
    min_origin_observations=DEFAULT_MIN_ORIGIN_OBSERVATIONS,
    min_origin_exits=DEFAULT_MIN_ORIGIN_EXITS,
    min_unique_values=DEFAULT_MIN_UNIQUE_VALUES,
):
    """Measure support for each state-specific macro deviation."""
    state_columns = _state_columns(base_transformer)
    states = [_state_from_column(column) for column in state_columns]
    reference_state = min(states)
    current_state = np.asarray(states)[
        np.argmax(X_train[state_columns].to_numpy(dtype=float), axis=1)
    ]
    next_state = y_train.to_numpy(dtype=int)
    records = []

    for main_term in market_terms:
        raw_column = base_transformer.term_spec(main_term).columns[0]
        values = X_train[raw_column].to_numpy(dtype=float)
        for state in states:
            mask = current_state == state
            state_values = values[mask]
            observations = int(mask.sum())
            exits = int(np.sum(next_state[mask] != state))
            unique_values = int(np.unique(state_values).size)
            eligible = (
                state != reference_state
                and observations >= int(min_origin_observations)
                and exits >= int(min_origin_exits)
                and unique_values >= int(min_unique_values)
            )
            record = {
                'MAIN_TERM': main_term,
                'INTERACTION_TERM': interaction_name(main_term),
                'ORIGIN_STATE': state,
                'REFERENCE_ORIGIN_STATE': reference_state,
                'OBSERVATIONS': observations,
                'EXITS': exits,
                'STAYS': observations - exits,
                'UNIQUE_MACRO_VALUES': unique_values,
                'MACRO_MIN': float(np.min(state_values)) if observations else np.nan,
                'MACRO_MAX': float(np.max(state_values)) if observations else np.nan,
                'ELIGIBLE_DEVIATION': bool(eligible),
                'MIN_ORIGIN_OBSERVATIONS': int(min_origin_observations),
                'MIN_ORIGIN_EXITS': int(min_origin_exits),
                'MIN_UNIQUE_VALUES': int(min_unique_values),
            }
            for destination in states:
                record[f'NEXT_STATE_{destination}_COUNT'] = int(
                    np.sum(next_state[mask] == destination)
                )
            records.append(record)
    return pd.DataFrame(records)


def interaction_state_map_from_support(support):
    state_map = {}
    if support.empty:
        return state_map
    for term_name, rows in support.groupby('INTERACTION_TERM', sort=False):
        states = rows.loc[rows['ELIGIBLE_DEVIATION'], 'ORIGIN_STATE'].astype(int).tolist()
        if states:
            state_map[str(term_name)] = states
    return state_map


class StateInteractionMLGTransformer:
    """Augment an additive MLG design with state-by-macro smooth deviations."""

    def __init__(self, base_transformer, base_terms, interaction_state_map):
        self.base_transformer = base_transformer
        self.base_terms = list(base_terms)
        self.interaction_state_map = {
            str(term): [int(state) for state in states]
            for term, states in interaction_state_map.items()
        }
        self.interaction_terms = list(self.interaction_state_map)
        self.feature_columns_ = list(base_transformer.feature_columns_)
        self.state_columns_ = _state_columns(base_transformer)
        self.state_column_map_ = {
            _state_from_column(column): column for column in self.state_columns_
        }

        if 'STATE_LAG_0' not in self.base_terms:
            raise ValueError('State-interaction MLG requires STATE_LAG_0 in the base model.')
        if any(term.startswith('DURATION') for term in self.base_terms):
            raise ValueError(
                'State-interaction MLG excludes all duration predictors.'
            )
        for term_name, states in self.interaction_state_map.items():
            main_term = interaction_main_term(term_name)
            if main_term not in self.base_terms:
                raise ValueError(
                    f'Interaction {term_name} violates hierarchy because {main_term} '
                    'is not a selected base term.'
                )
            missing_states = set(states) - set(self.state_column_map_)
            if missing_states:
                raise ValueError(
                    f'Interaction {term_name} has unknown states: {sorted(missing_states)}'
                )

    @property
    def term_names_(self):
        return self.base_terms + self.interaction_terms

    def transform_term(self, X, term_name, penalized=True):
        if term_name in self.base_terms:
            return self.base_transformer.transform_term(
                X, term_name, penalized=penalized
            )
        if term_name not in self.interaction_state_map:
            raise KeyError(f'Unknown state-interaction MLG term: {term_name}')

        main_term = interaction_main_term(term_name)
        main_basis = self.base_transformer.transform_term(
            X, main_term, penalized=penalized
        )
        matrices = []
        for state in self.interaction_state_map[term_name]:
            indicator = X[self.state_column_map_[state]].to_numpy(dtype=float)
            matrices.append(main_basis * indicator.reshape(-1, 1))
        return np.hstack(matrices)

    def transform(self, X, term_names=None, penalized=True):
        names = self.term_names_ if term_names is None else list(term_names)
        matrices = [
            self.transform_term(X, name, penalized=penalized) for name in names
        ]
        if not matrices:
            return np.empty((len(X), 0))
        return np.hstack(matrices)

    def term_slices(self, X, term_names=None):
        names = self.term_names_ if term_names is None else list(term_names)
        slices = {}
        start = 0
        for name in names:
            width = self.transform_term(X.iloc[:1], name).shape[1]
            slices[name] = slice(start, start + width)
            start += width
        return slices

    def term_spec(self, term_name):
        main_term = interaction_main_term(term_name)
        if main_term is not None:
            return self.base_transformer.term_spec(main_term)
        return self.base_transformer.term_spec(term_name)


@dataclass
class PreparedStateInteractionDesign:
    transformer: StateInteractionMLGTransformer
    base_terms: list
    selected_interactions: list
    base_p_values: dict
    interaction_p_values: dict
    interaction_fdr_p_values: dict
    interaction_methods: dict
    interaction_improvements: dict
    interaction_replicates: dict
    X_train: pd.DataFrame
    y_train: pd.Series

    @property
    def selected_terms(self):
        return self.base_terms + self.selected_interactions

    @property
    def term_p_values(self):
        values = dict(self.base_p_values)
        values.update({
            term: self.interaction_fdr_p_values.get(
                term, self.interaction_p_values.get(term, np.nan)
            )
            for term in self.selected_interactions
        })
        return values


class StateInteractionMLGTransitionModel:
    """Unweighted regularized MLG with selected current-state macro interactions."""

    def __init__(self, prepared_design, regularization_c, random_state=42):
        self.transformer = prepared_design.transformer
        self.base_terms = list(prepared_design.base_terms)
        self.selected_interactions = list(prepared_design.selected_interactions)
        self.selected_terms = list(prepared_design.selected_terms)
        self.term_p_values = dict(prepared_design.term_p_values)
        self.interaction_p_values = dict(prepared_design.interaction_p_values)
        self.interaction_fdr_p_values = dict(prepared_design.interaction_fdr_p_values)
        self.interaction_methods = dict(prepared_design.interaction_methods)
        self.regularization_c = float(regularization_c)
        self.inference_method = 'hierarchical_interaction_screen'
        self.classifier = LogisticRegression(
            C=self.regularization_c,
            penalty='l2',
            solver='lbfgs',
            max_iter=5000,
            random_state=random_state,
        )
        design = self.transformer.transform(
            prepared_design.X_train, self.selected_terms
        )
        self.classifier.fit(design, prepared_design.y_train)

    @property
    def classes_(self):
        return self.classifier.classes_

    def predict_proba(self, X):
        design = self.transformer.transform(X, self.selected_terms)
        return self.classifier.predict_proba(design)


def _conditional_wald_p_value(transformer, X_train, y_train, active_terms, term_name):
    design = transformer.transform(X_train, active_terms, penalized=False)
    result = _fit_mnlogit_for_inference(design, y_train)
    slices = transformer.term_slices(X_train, active_terms)
    return _joint_wald_p_values(result, {term_name: slices[term_name]})[term_name]


def _conditional_interaction_test(
    transformer,
    X_train,
    y_train,
    active_terms,
    term_name,
    alpha,
    inference_method,
    bootstrap_c,
    n_bootstrap,
    random_state,
):
    if inference_method not in {'auto', 'wald', 'regularized_bootstrap'}:
        raise ValueError(f'Unknown interaction inference method: {inference_method}')

    if inference_method in {'auto', 'wald'}:
        try:
            p_value = _conditional_wald_p_value(
                transformer, X_train, y_train, active_terms, term_name
            )
            return float(p_value), 'wald', np.nan, 0
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            if inference_method == 'wald':
                raise

    rng = np.random.default_rng(random_state)
    p_value, improvement, valid_count = _bootstrap_term_p_value(
        transformer,
        X_train,
        y_train,
        active_terms,
        term_name,
        regularization_c=bootstrap_c,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        rng=rng,
    )
    return float(p_value), 'regularized_bootstrap', float(improvement), int(valid_count)


def select_state_interactions(
    transformer,
    X_train,
    y_train,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Screen interaction blocks conditional on the fixed additive MLG base."""
    raw_p_values = {}
    methods = {}
    improvements = {}
    replicates = {}

    for index, term_name in enumerate(transformer.interaction_terms):
        active_terms = transformer.base_terms + [term_name]
        p_value, method, improvement, valid_count = _conditional_interaction_test(
            transformer,
            X_train,
            y_train,
            active_terms,
            term_name,
            alpha,
            inference_method,
            bootstrap_c,
            n_bootstrap,
            random_state + 1000 * (index + 1),
        )
        raw_p_values[term_name] = p_value
        methods[term_name] = method
        improvements[term_name] = improvement
        replicates[term_name] = valid_count

    adjusted = benjamini_hochberg(raw_p_values)
    selected = [
        term for term in transformer.interaction_terms
        if adjusted.get(term, 1.0) <= alpha
    ]

    # Re-test surviving terms conditionally on one another and remove unstable blocks.
    while len(selected) > 1:
        joint_raw = {}
        for index, term_name in enumerate(selected):
            p_value, method, improvement, valid_count = _conditional_interaction_test(
                transformer,
                X_train,
                y_train,
                transformer.base_terms + selected,
                term_name,
                alpha,
                inference_method,
                bootstrap_c,
                n_bootstrap,
                random_state + 100000 + 1000 * (index + 1),
            )
            joint_raw[term_name] = p_value
            methods[term_name] = method
            improvements[term_name] = improvement
            replicates[term_name] = valid_count
        joint_adjusted = benjamini_hochberg(joint_raw)
        retained = [
            term for term in selected if joint_adjusted.get(term, 1.0) <= alpha
        ]
        for term_name in selected:
            raw_p_values[term_name] = joint_raw[term_name]
            adjusted[term_name] = joint_adjusted[term_name]
        if retained == selected:
            break
        selected = retained

    return selected, raw_p_values, adjusted, methods, improvements, replicates


def run_duration_free_base_grid(
    n_states,
    exog_lags=(0, 1, 2),
    knot_grid=DEFAULT_KNOTS,
    regularization_grid=DEFAULT_REGULARIZATION_C,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Select an additive state-and-market MLG without duration predictors."""
    states, exog_rets = _read_transition_inputs(n_states)
    rows = []
    errors = []
    for exog_lag in exog_lags:
        X, y, meta, state_classes = build_transition_dataset(
            states,
            exog_rets,
            max_state_lag=0,
            max_exog_lag=exog_lag,
        )
        splits = split_transition_dataset(X, y, meta)
        X_train, y_train, _ = splits['train']
        duration_free_train = duration_free_feature_frame(X_train)

        for n_knots in knot_grid:
            try:
                prepared = prepare_mlg_design(
                    duration_free_train,
                    y_train,
                    n_knots=n_knots,
                    alpha=alpha,
                    inference_method=inference_method,
                    bootstrap_c=bootstrap_c,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                )
                if 'STATE_LAG_0' not in prepared.selected_terms:
                    prepared.selected_terms.insert(0, 'STATE_LAG_0')
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                errors.append({
                    'N_STATES': n_states,
                    'EXOG_LAG': exog_lag,
                    'N_KNOTS': n_knots,
                    'ERROR': str(error),
                })
                continue

            if any(term.startswith('DURATION') for term in prepared.selected_terms):
                raise AssertionError('Duration entered the duration-free SI-MLG base.')
            optional_terms = [
                term for term in prepared.selected_terms
                if term != 'STATE_LAG_0'
            ]
            optional_significant = all(
                prepared.term_p_values.get(term, 1.0) <= alpha
                for term in optional_terms
            )
            common = {
                'N_STATES': n_states,
                'STATE_LAG': 0,
                'EXOG_LAG': int(exog_lag),
                'N_KNOTS': int(n_knots),
                'SIGNIFICANCE_LEVEL': float(alpha),
                'MODEL_SPECIFICATION': MODEL_SPECIFICATION,
                'DURATION_EXCLUDED': True,
                'REQUIRED_TERMS': 'STATE_LAG_0',
                'SELECTED_TERM_COUNT': len(prepared.selected_terms),
                'SELECTED_TERMS': '|'.join(prepared.selected_terms),
                'TERM_P_VALUES': json.dumps(
                    prepared.term_p_values, sort_keys=True
                ),
                'STATE_TERM_P_VALUE': prepared.term_p_values.get(
                    'STATE_LAG_0', np.nan
                ),
                'ALL_OPTIONAL_TERMS_SIGNIFICANT': optional_significant,
                'INFERENCE_METHOD': prepared.inference_method,
                'BOOTSTRAP_C': float(bootstrap_c),
                'BOOTSTRAP_REPLICATES': (
                    int(n_bootstrap)
                    if prepared.inference_method == 'regularized_bootstrap'
                    else 0
                ),
                'TERM_BOOTSTRAP_IMPROVEMENTS': json.dumps(
                    prepared.term_improvements or {}, sort_keys=True
                ),
                'TERM_BOOTSTRAP_VALID_REPLICATES': json.dumps(
                    prepared.bootstrap_replicates or {}, sort_keys=True
                ),
                'TRAIN_ROWS': len(X_train),
            }
            for regularization_c in regularization_grid:
                model = MLGTransitionModel(prepared, regularization_c)
                for split_name in ['validation', 'prediction']:
                    X_eval, y_eval, _ = splits[split_name]
                    probabilities = _align_probabilities(
                        model.predict_proba(
                            duration_free_feature_frame(X_eval)
                        ),
                        model.classes_,
                        state_classes,
                    )
                    extra = dict(common)
                    extra.update({
                        'SPLIT': split_name,
                        'SPLIT_START': X_eval.index.min().date(),
                        'SPLIT_END': X_eval.index.max().date(),
                        'REGULARIZATION_C': float(regularization_c),
                        'SMOOTHING_LAMBDA': 1.0 / float(regularization_c),
                        'EVAL_ROWS': len(X_eval),
                    })
                    rows.append(evaluate_probabilities(
                        BASE_MODEL_NAME,
                        y_eval,
                        probabilities,
                        state_classes,
                        extra,
                    ))

    error_path = (
        BASE_DIR
        / 'phase_1'
        / f'state_interaction_mlg_base_errors_q{n_states}.csv'
    )
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    if not rows:
        raise RuntimeError(
            f'Every duration-free SI-MLG base candidate failed for q{n_states}.'
        )
    return pd.DataFrame(rows)


def best_duration_free_base_config(report):
    candidates = report[
        (report['MODEL'] == BASE_MODEL_NAME)
        & (report['SPLIT'] == 'validation')
    ].copy()
    if candidates.empty:
        raise ValueError('No duration-free SI-MLG base candidate is available.')
    return candidates.sort_values(
        [
            'LOG_LOSS',
            'BRIER_SCORE',
            'ACCURACY',
            'SELECTED_TERM_COUNT',
            'EXOG_LAG',
            'N_KNOTS',
            'REGULARIZATION_C',
        ],
        ascending=[True, True, False, True, True, True, True],
    ).iloc[0]


def save_duration_free_base_report(n_states, **grid_options):
    report = run_duration_free_base_grid(n_states, **grid_options)
    report_path = (
        BASE_DIR
        / 'phase_1'
        / f'state_interaction_mlg_base_report_q{n_states}.csv'
    )
    report.to_csv(report_path, index=False)
    best = best_duration_free_base_config(report)
    p_values = _json_dict(best['TERM_P_VALUES'])
    selected = set(_pipe_list(best['SELECTED_TERMS']))
    significance = pd.DataFrame([
        {
            'N_STATES': n_states,
            'TERM': term,
            'TERM_GROUP': (
                'state_required' if term == 'STATE_LAG_0'
                else 'market_main'
            ),
            'P_VALUE': p_value,
            'SELECTED': term in selected,
            'SIGNIFICANT_AT_5_PERCENT': p_value <= SIGNIFICANCE_LEVEL,
            'DURATION_EXCLUDED': True,
        }
        for term, p_value in p_values.items()
    ])
    significance.to_csv(
        BASE_DIR
        / 'phase_1'
        / f'state_interaction_mlg_base_significance_q{n_states}.csv',
        index=False,
    )
    return report, report_path


def load_or_create_duration_free_base_report(n_states):
    path = (
        BASE_DIR
        / 'phase_1'
        / f'state_interaction_mlg_base_report_q{n_states}.csv'
    )
    if path.exists():
        report = pd.read_csv(path)
        required = {
            'MODEL_SPECIFICATION',
            'DURATION_EXCLUDED',
            'SELECTED_TERMS',
            'TERM_P_VALUES',
        }
        if required.issubset(report.columns):
            specification = report['MODEL_SPECIFICATION'].astype(str)
            no_duration = report['DURATION_EXCLUDED'].astype(
                str
            ).str.lower().eq('true')
            if specification.eq(MODEL_SPECIFICATION).all() and no_duration.all():
                return report
    report, _ = save_duration_free_base_report(n_states)
    return report


def _base_design_from_selected_duration_free_mlg(n_states):
    additive_report = load_or_create_duration_free_base_report(n_states)
    additive_best = best_duration_free_base_config(additive_report)
    states, exog_rets = _read_transition_inputs(n_states)
    X, y, meta, state_classes = build_transition_dataset(
        states,
        exog_rets,
        max_state_lag=0,
        max_exog_lag=int(additive_best['EXOG_LAG']),
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']
    base_transformer = MLGFeatureTransformer(
        n_knots=int(additive_best['N_KNOTS'])
    ).fit(duration_free_feature_frame(X_train))
    base_terms = _pipe_list(additive_best['SELECTED_TERMS'])
    missing = set(base_terms) - set(base_transformer.term_names_)
    if missing:
        raise ValueError(
            f'Selected duration-free base terms are unavailable: {sorted(missing)}'
        )
    if 'STATE_LAG_0' not in base_terms:
        raise ValueError('The duration-free SI-MLG base must include STATE_LAG_0.')
    if any(term.startswith('DURATION') for term in base_terms):
        raise ValueError('The duration-free SI-MLG base contains a duration term.')
    base_p_values = _json_dict(additive_best['TERM_P_VALUES'])
    return (
        additive_best,
        X,
        y,
        meta,
        state_classes,
        splits,
        base_transformer,
        base_terms,
        base_p_values,
    )


def prepare_state_interaction_design(
    n_states,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
    min_origin_observations=DEFAULT_MIN_ORIGIN_OBSERVATIONS,
    min_origin_exits=DEFAULT_MIN_ORIGIN_EXITS,
    min_unique_values=DEFAULT_MIN_UNIQUE_VALUES,
):
    (
        additive_best,
        X,
        y,
        meta,
        state_classes,
        splits,
        base_transformer,
        base_terms,
        base_p_values,
    ) = _base_design_from_selected_duration_free_mlg(n_states)
    X_train, y_train, _ = splits['train']
    market_terms = [term for term in base_terms if is_market_term(term)]
    support = interaction_support_table(
        X_train,
        y_train,
        base_transformer,
        market_terms,
        min_origin_observations=min_origin_observations,
        min_origin_exits=min_origin_exits,
        min_unique_values=min_unique_values,
    )
    state_map = interaction_state_map_from_support(support)
    transformer = StateInteractionMLGTransformer(
        base_transformer,
        base_terms,
        state_map,
    )
    (
        selected_interactions,
        raw_p_values,
        adjusted_p_values,
        methods,
        improvements,
        replicates,
    ) = select_state_interactions(
        transformer,
        X_train,
        y_train,
        alpha=alpha,
        inference_method=inference_method,
        bootstrap_c=bootstrap_c,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    prepared = PreparedStateInteractionDesign(
        transformer=transformer,
        base_terms=base_terms,
        selected_interactions=selected_interactions,
        base_p_values=base_p_values,
        interaction_p_values=raw_p_values,
        interaction_fdr_p_values=adjusted_p_values,
        interaction_methods=methods,
        interaction_improvements=improvements,
        interaction_replicates=replicates,
        X_train=X_train,
        y_train=y_train,
    )
    return prepared, support, additive_best, splits, state_classes


def run_state_interaction_mlg_grid(
    n_states,
    regularization_grid=DEFAULT_REGULARIZATION_C,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
    min_origin_observations=DEFAULT_MIN_ORIGIN_OBSERVATIONS,
    min_origin_exits=DEFAULT_MIN_ORIGIN_EXITS,
    min_unique_values=DEFAULT_MIN_UNIQUE_VALUES,
):
    prepared, support, additive_best, splits, state_classes = (
        prepare_state_interaction_design(
            n_states,
            alpha=alpha,
            inference_method=inference_method,
            bootstrap_c=bootstrap_c,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            min_origin_observations=min_origin_observations,
            min_origin_exits=min_origin_exits,
            min_unique_values=min_unique_values,
        )
    )
    interaction_map = {
        term: prepared.transformer.interaction_state_map[term]
        for term in prepared.selected_interactions
    }
    common = {
        'N_STATES': n_states,
        'STATE_LAG': 0,
        'EXOG_LAG': int(additive_best['EXOG_LAG']),
        'N_KNOTS': int(additive_best['N_KNOTS']),
        'SIGNIFICANCE_LEVEL': float(alpha),
        'MODEL_SPECIFICATION': MODEL_SPECIFICATION,
        'DURATION_EXCLUDED': True,
        'BASE_MODEL': BASE_MODEL_NAME,
        'BASE_REGULARIZATION_C': float(additive_best['REGULARIZATION_C']),
        'SELECTED_BASE_TERM_COUNT': len(prepared.base_terms),
        'SELECTED_BASE_TERMS': '|'.join(prepared.base_terms),
        'CANDIDATE_INTERACTION_COUNT': len(prepared.transformer.interaction_terms),
        'SELECTED_INTERACTION_COUNT': len(prepared.selected_interactions),
        'SELECTED_INTERACTIONS': '|'.join(prepared.selected_interactions),
        'SELECTED_TERM_COUNT': len(prepared.selected_terms),
        'SELECTED_TERMS': '|'.join(prepared.selected_terms),
        'INTERACTION_P_VALUES': json.dumps(
            prepared.interaction_p_values, sort_keys=True
        ),
        'INTERACTION_FDR_P_VALUES': json.dumps(
            prepared.interaction_fdr_p_values, sort_keys=True
        ),
        'INTERACTION_METHODS': json.dumps(
            prepared.interaction_methods, sort_keys=True
        ),
        'INTERACTION_BOOTSTRAP_IMPROVEMENTS': json.dumps(
            prepared.interaction_improvements, sort_keys=True
        ),
        'INTERACTION_BOOTSTRAP_VALID_REPLICATES': json.dumps(
            prepared.interaction_replicates, sort_keys=True
        ),
        'INTERACTION_STATE_MAP': json.dumps(interaction_map, sort_keys=True),
        'FDR_METHOD': 'Benjamini-Hochberg',
        'ALL_INTERACTIONS_SIGNIFICANT': all(
            prepared.interaction_fdr_p_values.get(term, 1.0) <= alpha
            for term in prepared.selected_interactions
        ),
        'LOSS_FUNCTION': 'unweighted multinomial negative log likelihood plus standard L2/P-spline penalty',
        'TRAIN_ROWS': len(prepared.X_train),
        'MIN_ORIGIN_OBSERVATIONS': int(min_origin_observations),
        'MIN_ORIGIN_EXITS': int(min_origin_exits),
        'MIN_UNIQUE_VALUES': int(min_unique_values),
    }
    rows = []
    for regularization_c in regularization_grid:
        model = StateInteractionMLGTransitionModel(
            prepared,
            regularization_c=regularization_c,
            random_state=random_state,
        )
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
                'REGULARIZATION_C': float(regularization_c),
                'SMOOTHING_LAMBDA': 1.0 / float(regularization_c),
                'EVAL_ROWS': len(X_eval),
            })
            rows.append(evaluate_probabilities(
                MODEL_NAME,
                y_eval,
                probabilities,
                state_classes,
                extra,
            ))
    return pd.DataFrame(rows), support


def best_state_interaction_mlg_config(report):
    candidates = report[
        (report['MODEL'] == MODEL_NAME)
        & (report['SPLIT'] == 'validation')
    ].copy()
    if candidates.empty:
        raise ValueError('No state-interaction MLG validation candidate is available.')
    return candidates.sort_values(
        [
            'LOG_LOSS',
            'BRIER_SCORE',
            'ACCURACY',
            'SELECTED_INTERACTION_COUNT',
            'REGULARIZATION_C',
        ],
        ascending=[True, True, False, True, True],
    ).iloc[0]


def save_state_interaction_mlg_transition_report(n_states, **grid_options):
    report, support = run_state_interaction_mlg_grid(n_states, **grid_options)
    report_path = (
        BASE_DIR / 'phase_1' / f'state_interaction_mlg_transition_report_q{n_states}.csv'
    )
    report.to_csv(report_path, index=False)
    support.to_csv(
        BASE_DIR / 'phase_1' / f'state_interaction_mlg_support_q{n_states}.csv',
        index=False,
    )

    best = best_state_interaction_mlg_config(report)
    raw = json.loads(best['INTERACTION_P_VALUES'])
    adjusted = json.loads(best['INTERACTION_FDR_P_VALUES'])
    methods = json.loads(best['INTERACTION_METHODS'])
    selected = set(_pipe_list(best['SELECTED_INTERACTIONS']))
    significance = pd.DataFrame([
        {
            'N_STATES': n_states,
            'INTERACTION_TERM': term,
            'MAIN_TERM': interaction_main_term(term),
            'RAW_P_VALUE': raw[term],
            'FDR_P_VALUE': adjusted[term],
            'INFERENCE_METHOD': methods[term],
            'SELECTED': term in selected,
            'SIGNIFICANT_AT_5_PERCENT_FDR': adjusted[term] <= SIGNIFICANCE_LEVEL,
        }
        for term in raw
    ])
    significance.to_csv(
        BASE_DIR
        / 'phase_1'
        / f'state_interaction_mlg_selected_significance_q{n_states}.csv',
        index=False,
    )
    return report, report_path


def plot_state_interaction_significance(n_states, output_dir):
    """Plot conditional raw and FDR-adjusted interaction p-values."""
    significance_path = (
        BASE_DIR
        / 'phase_1'
        / f'state_interaction_mlg_selected_significance_q{n_states}.csv'
    )
    significance = pd.read_csv(significance_path).sort_values(
        ['FDR_P_VALUE', 'RAW_P_VALUE'],
        ascending=[False, False],
    )
    labels = significance['MAIN_TERM'].fillna(
        significance['INTERACTION_TERM']
    )
    positions = np.arange(len(significance))
    height = max(4.4, 0.34 * len(significance) + 1.7)
    fig, ax = plt.subplots(figsize=(9.5, height), constrained_layout=True)
    ax.scatter(
        significance['RAW_P_VALUE'],
        positions - 0.11,
        label='Conditional raw p-value',
        color='#D17A22',
        marker='o',
        s=42,
        zorder=3,
    )
    ax.scatter(
        significance['FDR_P_VALUE'],
        positions + 0.11,
        label='Benjamini-Hochberg FDR p-value',
        color='#3C6E71',
        marker='D',
        s=38,
        zorder=3,
    )
    ax.axvline(
        SIGNIFICANCE_LEVEL,
        color='#A63D40',
        linestyle='--',
        linewidth=1.4,
        label='5% selection threshold',
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(left=0)
    ax.set_xlabel('p-value')
    ax.set_title(
        f'q{n_states} state-by-market interaction selection evidence'
    )
    ax.grid(axis='x', alpha=0.25)
    ax.legend(loc='lower right', fontsize=8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'state_interaction_significance_q{n_states}.png'
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return path


def load_or_create_state_interaction_mlg_report(n_states):
    path = (
        BASE_DIR / 'phase_1' / f'state_interaction_mlg_transition_report_q{n_states}.csv'
    )
    if path.exists():
        report = pd.read_csv(path)
        required = {
            'MODEL_SPECIFICATION',
            'DURATION_EXCLUDED',
            'INTERACTION_STATE_MAP',
            'SELECTED_INTERACTIONS',
            'REGULARIZATION_C',
        }
        if required.issubset(report.columns):
            specification = report['MODEL_SPECIFICATION'].astype(str)
            no_duration = report['DURATION_EXCLUDED'].astype(
                str
            ).str.lower().eq('true')
            if specification.eq(MODEL_SPECIFICATION).all() and no_duration.all():
                return report
    report, _ = save_state_interaction_mlg_transition_report(n_states)
    return report


def _selected_design_from_report(n_states, best):
    (
        additive_best,
        X,
        y,
        meta,
        state_classes,
        splits,
        base_transformer,
        base_terms,
        base_p_values,
    ) = _base_design_from_selected_duration_free_mlg(n_states)
    selected_interactions = _pipe_list(best.get('SELECTED_INTERACTIONS', ''))
    selected_state_map = _json_dict(best.get('INTERACTION_STATE_MAP', '{}'))
    transformer = StateInteractionMLGTransformer(
        base_transformer,
        base_terms,
        selected_state_map,
    )
    raw = _json_dict(best.get('INTERACTION_P_VALUES', '{}'))
    adjusted = _json_dict(best.get('INTERACTION_FDR_P_VALUES', '{}'))
    methods = _json_dict(best.get('INTERACTION_METHODS', '{}'))
    improvements = _json_dict(
        best.get('INTERACTION_BOOTSTRAP_IMPROVEMENTS', '{}')
    )
    replicates = _json_dict(
        best.get('INTERACTION_BOOTSTRAP_VALID_REPLICATES', '{}')
    )
    X_train, y_train, _ = splits['train']
    prepared = PreparedStateInteractionDesign(
        transformer=transformer,
        base_terms=base_terms,
        selected_interactions=selected_interactions,
        base_p_values=base_p_values,
        interaction_p_values=raw,
        interaction_fdr_p_values=adjusted,
        interaction_methods=methods,
        interaction_improvements=improvements,
        interaction_replicates=replicates,
        X_train=X_train,
        y_train=y_train,
    )
    return prepared, splits, state_classes, additive_best


def fit_selected_state_interaction_mlg_transition(n_states, report=None):
    report = (
        load_or_create_state_interaction_mlg_report(n_states)
        if report is None else report
    )
    best = best_state_interaction_mlg_config(report)
    prepared, splits, state_classes, _ = _selected_design_from_report(n_states, best)
    model = StateInteractionMLGTransitionModel(
        prepared,
        regularization_c=float(best['REGULARIZATION_C']),
    )
    return model, splits, state_classes, best


def save_selected_state_interaction_probabilities(
    n_states, model, splits, state_classes, output_dir
):
    X_prediction, y_prediction, meta_prediction = splits['prediction']
    probabilities = _align_probabilities(
        model.predict_proba(X_prediction), model.classes_, state_classes
    )
    frame = pd.DataFrame({
        'DATE': X_prediction.index,
        'CURRENT_STATE': meta_prediction['CURRENT_STATE'].to_numpy(dtype=int),
        'ACTUAL_NEXT_STATE': y_prediction.to_numpy(dtype=int),
    })
    for column, state in enumerate(state_classes):
        frame[f'SI_MLG_P_TO_{state}'] = probabilities[:, column]
    frame['SI_MLG_ROW_SUM'] = probabilities.sum(axis=1)
    frame['SI_MLG_PREDICTED_NEXT_STATE'] = [
        state_classes[index] for index in probabilities.argmax(axis=1)
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'state_interaction_mlg_probabilities_q{n_states}.csv'
    frame.to_csv(path, index=False)
    return path


def plot_state_interaction_feature_importance(
    n_states, model, X_train, output_dir
):
    """Save coefficient-norm importance for base and interaction terms."""
    slices = model.transformer.term_slices(X_train, model.selected_terms)
    coefficients = np.asarray(model.classifier.coef_)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    records = []
    for term_name in model.selected_terms:
        term_slice = slices[term_name]
        main_term = interaction_main_term(term_name)
        if main_term is not None:
            group = 'state_macro_interaction'
        elif term_name.startswith('STATE_LAG_'):
            group = 'state'
        elif term_name.startswith('DURATION'):
            raise AssertionError(
                'Duration must not appear in SI-MLG feature importance.'
            )
        else:
            group = 'market_main'
        records.append({
            'N_STATES': n_states,
            'MODEL': 'STATE_INTERACTION_MLG',
            'TERM': term_name,
            'TERM_GROUP': group,
            'MAIN_TERM': main_term,
            'IMPORTANCE': float(np.linalg.norm(coefficients[:, term_slice])),
            'TERM_P_VALUE': model.term_p_values.get(term_name),
            'REGULARIZATION_C': model.regularization_c,
            'COEFFICIENT_SCALE': 'penalized_basis_coefficients',
        })
    frame = pd.DataFrame(records)
    total = frame['IMPORTANCE'].sum()
    frame['NORMALIZED_IMPORTANCE'] = (
        frame['IMPORTANCE'] / total if total > 0 else 0.0
    )
    frame = frame.sort_values('NORMALIZED_IMPORTANCE', ascending=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'feature_importance_state_interaction_mlg_q{n_states}.csv'
    frame.to_csv(csv_path, index=False)

    height = max(4.2, 0.32 * len(frame) + 1.6)
    fig, ax = plt.subplots(figsize=(9.2, height), constrained_layout=True)
    colors = {
        'state': '#3C6E71',
        'market_main': '#D17A22',
        'duration': '#8D5A97',
        'state_macro_interaction': '#A63D40',
    }
    ax.barh(
        frame['TERM'],
        frame['NORMALIZED_IMPORTANCE'],
        color=[colors[group] for group in frame['TERM_GROUP']],
    )
    ax.set_xlabel('Normalized coefficient-norm importance')
    ax.set_title(f'q{n_states} selected state-interaction MLG importance')
    present_groups = list(dict.fromkeys(frame['TERM_GROUP']))
    handles = [
        Patch(
            facecolor=colors[group],
            label=group.replace('_', ' ').title(),
        )
        for group in present_groups
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    path = output_dir / f'feature_importance_state_interaction_mlg_q{n_states}.png'
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return path, csv_path


def _counterfactual_log_odds(
    model, sample_X, raw_column, values, numerator, denominator
):
    classes = np.asarray(model.classes_)
    numerator_index = int(np.where(classes == numerator)[0][0])
    denominator_index = int(np.where(classes == denominator)[0][0])
    fitted = []
    for value in values:
        counterfactual = sample_X.copy()
        counterfactual.loc[:, raw_column] = value
        probabilities = model.predict_proba(counterfactual)
        fitted.append(float(np.mean(np.log(
            np.clip(probabilities[:, numerator_index], 1e-12, 1.0)
            / np.clip(probabilities[:, denominator_index], 1e-12, 1.0)
        ))))
    return np.asarray(fitted)


def plot_state_specific_macro_effects(
    n_states,
    model,
    X_train,
    y_train,
    meta_train,
    output_dir,
):
    """Plot observed and fitted macro effects separately by current origin state."""
    output_dir = (
        Path(output_dir) / f'state_interaction_mlg_partial_effects_q{n_states}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_classes = [int(state) for state in model.classes_]
    records = []
    paths = []

    market_terms = [term for term in model.base_terms if is_market_term(term)]
    selected_interactions = set(model.selected_interactions)
    for main_term in market_terms:
        raw_column = model.transformer.base_transformer.term_spec(
            main_term
        ).columns[0]
        interaction = interaction_name(main_term)
        interaction_selected = interaction in selected_interactions

        for origin_state in state_classes:
            mask = (
                meta_train['CURRENT_STATE'].reindex(X_train.index).to_numpy(dtype=int)
                == origin_state
            )
            sample_X = X_train.loc[mask].copy()
            sample_y = y_train.loc[sample_X.index].to_numpy(dtype=int)
            if sample_X.empty:
                continue
            observed_values = sample_X[raw_column].to_numpy(dtype=float)
            unique_values = np.unique(observed_values)
            if unique_values.size <= 60:
                grid = unique_values
            else:
                lower, upper = np.quantile(observed_values, [0.01, 0.99])
                grid = np.linspace(lower, upper, 60)

            for destination_state in state_classes:
                if destination_state == origin_state:
                    continue
                numerator = origin_state
                denominator = destination_state
                fitted = _counterfactual_log_odds(
                    model,
                    sample_X,
                    raw_column,
                    grid,
                    numerator,
                    denominator,
                )
                observed = _binned_empirical_log_odds(
                    observed_values,
                    sample_y,
                    numerator,
                    denominator,
                    categorical=False,
                )

                fig, ax = plt.subplots(figsize=(7.5, 4.7), constrained_layout=True)
                ax.plot(
                    grid,
                    fitted,
                    color='#1f5a85',
                    linewidth=2.2,
                    label='Fitted SI-MLG state-specific curve',
                )
                if not observed.empty:
                    ax.errorbar(
                        observed['PREDICTOR_VALUE'],
                        observed['OBSERVED_LOG_ODDS'],
                        yerr=1.96 * observed['OBSERVED_LOG_ODDS_SE'],
                        fmt='o',
                        color='#b04a24',
                        ecolor='#b04a24',
                        capsize=3,
                        label='Observed binned log-odds (95% approximate CI)',
                    )
                ax.axhline(0, color='black', linewidth=1, linestyle='--')
                interaction_text = (
                    'selected state interaction'
                    if interaction_selected else 'pooled macro effect'
                )
                ax.set_title(
                    f'q{n_states} SI-MLG: {main_term}, origin {origin_state}; '
                    f'{origin_state} vs {destination_state}\n{interaction_text}'
                )
                ax.set_xlabel(main_term)
                ax.set_ylabel(
                    f'log[P(next={origin_state}) / P(next={destination_state})]'
                )
                ax.legend(loc='best', fontsize=8)
                safe_term = re.sub(r'[^A-Za-z0-9_.-]+', '_', main_term)
                path = (
                    output_dir
                    / f'{safe_term}__origin_{origin_state}__'
                    f'{origin_state}_vs_{destination_state}.png'
                )
                fig.savefig(path, dpi=160, bbox_inches='tight')
                plt.close(fig)
                paths.append(path)

                common = {
                    'N_STATES': n_states,
                    'MAIN_TERM': main_term,
                    'INTERACTION_TERM': interaction,
                    'INTERACTION_SELECTED': interaction_selected,
                    'INTERACTION_FDR_P_VALUE': model.interaction_fdr_p_values.get(
                        interaction
                    ),
                    'ORIGIN_STATE': origin_state,
                    'LOG_ODDS_CONTRAST': (
                        f'{origin_state} vs {destination_state}'
                    ),
                }
                for value, fitted_value in zip(grid, fitted):
                    records.append({
                        **common,
                        'ROW_TYPE': 'FITTED_CURVE',
                        'PREDICTOR_VALUE': value,
                        'FITTED_LOG_ODDS': fitted_value,
                    })
                for row in observed.to_dict('records'):
                    records.append({
                        **common,
                        'ROW_TYPE': 'OBSERVED_BIN',
                        **row,
                    })

    csv_path = output_dir / 'state_specific_macro_partial_effects.csv'
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return paths, csv_path


def main():
    output_dir = BASE_DIR / 'phase_1' / 'transition_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    for n_states in [2, 4]:
        report, report_path = save_state_interaction_mlg_transition_report(n_states)
        model, splits, state_classes, best = (
            fit_selected_state_interaction_mlg_transition(n_states, report=report)
        )
        X_train, y_train, meta_train = splits['train']
        probability_path = save_selected_state_interaction_probabilities(
            n_states, model, splits, state_classes, output_dir
        )
        importance_path, importance_csv = plot_state_interaction_feature_importance(
            n_states, model, X_train, output_dir
        )
        significance_path = plot_state_interaction_significance(
            n_states, output_dir
        )
        partial_paths, partial_csv = plot_state_specific_macro_effects(
            n_states,
            model,
            X_train,
            y_train,
            meta_train,
            output_dir,
        )
        print(f'q{n_states} report: {report_path}')
        print(
            f'q{n_states} selected interactions: '
            f"{best['SELECTED_INTERACTIONS'] or '(none)'}"
        )
        print(f'q{n_states} probabilities: {probability_path}')
        print(f'q{n_states} importance: {importance_path} and {importance_csv}')
        print(f'q{n_states} significance: {significance_path}')
        print(f'q{n_states} partial effects: {partial_csv} ({len(partial_paths)} plots)')


if __name__ == '__main__':
    main()
