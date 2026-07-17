"""Regularized multinomial logistic GAM for HMM regime transitions."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from p1_3_transition_features import build_transition_dataset
from p1_4_dynamic_transition import (
    _align_probabilities,
    evaluate_probabilities,
    split_transition_dataset,
)
from project_config import BASE_DIR


MODEL_NAME = 'multinomial_logistic_gam'
SIGNIFICANCE_LEVEL = 0.05
DEFAULT_KNOTS = (4, 6)
DEFAULT_REGULARIZATION_C = (0.01, 0.1, 1.0, 10.0)
DEFAULT_BOOTSTRAP_REPLICATES = 49
DEFAULT_BOOTSTRAP_C = 1.0
BOOTSTRAP_RANDOM_STATE = 42


@dataclass
class _TermSpec:
    name: str
    columns: list
    kind: str
    transformer: object = None
    basis_map: object = None
    mean: float = 0.0
    scale: float = 1.0
    log_duration: bool = False


class MLGFeatureTransformer:
    """Build identifiable linear state terms and penalized spline terms."""

    def __init__(self, n_knots=4, degree=3, ridge_floor=0.01):
        self.n_knots = int(n_knots)
        self.degree = int(degree)
        self.ridge_floor = float(ridge_floor)
        self.terms_ = []
        self.feature_columns_ = None

    @property
    def term_names_(self):
        return [term.name for term in self.terms_]

    def fit(self, X):
        X = self._validate_frame(X)
        self.feature_columns_ = list(X.columns)
        self.terms_ = []
        has_state_specific_duration = any(
            column.startswith('DURATION_STATE_') for column in X.columns
        )

        state_groups = {}
        for column in X.columns:
            match = re.match(r'^(STATE_LAG_\d+)_', column)
            if match:
                state_groups.setdefault(match.group(1), []).append(column)

        grouped_columns = {column for columns in state_groups.values() for column in columns}
        for name, columns in state_groups.items():
            # Drop one state indicator per lag so the model is identifiable with an intercept.
            self.terms_.append(_TermSpec(name=name, columns=columns, kind='categorical'))

        for column in X.columns:
            if column in grouped_columns:
                continue
            if column == 'DURATION' and has_state_specific_duration:
                # The global duration basis is redundant with the state-specific duration bases.
                # Keeping all of them makes the unpenalized Wald-test design rank deficient.
                continue
            values = X[column].to_numpy(dtype=float)
            unique_count = np.unique(values).size
            log_duration = column.startswith('DURATION') and np.nanmin(values) >= 0
            transformed_values = np.log1p(values) if log_duration else values

            if unique_count < max(4, self.n_knots):
                mean = float(np.mean(transformed_values))
                scale = float(np.std(transformed_values)) or 1.0
                self.terms_.append(_TermSpec(
                    name=column,
                    columns=[column],
                    kind='linear',
                    mean=mean,
                    scale=scale,
                    log_duration=log_duration,
                ))
                continue

            spline = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                knots='uniform',
                include_bias=False,
                extrapolation='linear',
            )
            raw_basis = spline.fit_transform(transformed_values.reshape(-1, 1))
            width = raw_basis.shape[1]
            if width >= 3:
                second_difference = np.diff(np.eye(width), n=2, axis=0)
                penalty = second_difference.T @ second_difference
            else:
                penalty = np.zeros((width, width))

            # Whitening turns sklearn's L2 penalty into a P-spline roughness penalty.
            penalty = penalty + self.ridge_floor * np.eye(width)
            eigenvalues, eigenvectors = np.linalg.eigh(penalty)
            basis_map = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            self.terms_.append(_TermSpec(
                name=column,
                columns=[column],
                kind='spline',
                transformer=spline,
                basis_map=basis_map,
                log_duration=log_duration,
            ))

        return self

    def transform_term(self, X, term_name, penalized=True):
        X = self._validate_frame(X)
        term = self._term(term_name)

        if term.kind == 'categorical':
            # The final level is the reference category represented by all zeros.
            return X[term.columns[:-1]].to_numpy(dtype=float)

        values = X[term.columns[0]].to_numpy(dtype=float)
        if term.log_duration:
            values = np.log1p(np.clip(values, 0, None))

        if term.kind == 'linear':
            return ((values - term.mean) / term.scale).reshape(-1, 1)

        raw_basis = term.transformer.transform(values.reshape(-1, 1))
        if not penalized:
            return raw_basis
        return raw_basis @ term.basis_map

    def transform(self, X, term_names=None, penalized=True):
        names = self.term_names_ if term_names is None else list(term_names)
        matrices = [self.transform_term(X, name, penalized=penalized) for name in names]
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
        return self._term(term_name)

    def _term(self, term_name):
        for term in self.terms_:
            if term.name == term_name:
                return term
        raise KeyError(f'Unknown MLG term: {term_name}')

    def _validate_frame(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('MLG features must be provided as a pandas DataFrame.')
        if self.feature_columns_ is not None:
            missing = set(self.feature_columns_) - set(X.columns)
            if missing:
                raise ValueError(f'MLG features are missing columns: {sorted(missing)}')
            return X[self.feature_columns_]
        return X


def _fit_mnlogit_for_inference(design, y):
    columns = [f'x{index}' for index in range(design.shape[1])]
    exog = pd.DataFrame(design, index=y.index, columns=columns)
    exog.insert(0, 'const', 1.0)

    last_error = None
    for method, maxiter in [('newton', 200), ('lbfgs', 1000), ('bfgs', 1000)]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = sm.MNLogit(y, exog, check_rank=True).fit(
                    method=method,
                    maxiter=maxiter,
                    disp=False,
                    full_output=True,
                )
            converged = result.mle_retvals.get('converged', True)
            finite = np.isfinite(result.params.to_numpy()).all()
            finite = finite and np.isfinite(result.cov_params().to_numpy()).all()
            if converged and finite:
                return result
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            last_error = error

    detail = f' Last error: {last_error}' if last_error is not None else ''
    raise RuntimeError(f'The unpenalized MNLogit inference fit did not converge.{detail}')


def _joint_wald_p_values(result, term_slices):
    parameter_count = result.params.shape[0]
    equation_count = result.params.shape[1]
    total_count = parameter_count * equation_count
    p_values = {}

    for term_name, term_slice in term_slices.items():
        # Add one for the intercept inserted ahead of the transformed features.
        feature_positions = range(term_slice.start + 1, term_slice.stop + 1)
        positions = [
            equation * parameter_count + position
            for equation in range(equation_count)
            for position in feature_positions
        ]
        restrictions = np.zeros((len(positions), total_count))
        restrictions[np.arange(len(positions)), positions] = 1.0
        test = result.wald_test(restrictions, scalar=True)
        p_values[term_name] = float(np.asarray(test.pvalue).squeeze())

    return p_values


def select_significant_terms_wald(transformer, X_train, y_train, alpha=SIGNIFICANCE_LEVEL):
    """Backward-eliminate whole terms using train-only joint Wald tests."""
    active_terms = list(transformer.term_names_)
    last_p_values = {}

    while active_terms:
        design = transformer.transform(X_train, active_terms, penalized=False)
        result = _fit_mnlogit_for_inference(design, y_train)
        term_slices = transformer.term_slices(X_train, active_terms)
        p_values = _joint_wald_p_values(result, term_slices)
        last_p_values.update(p_values)

        worst_term = max(p_values, key=lambda name: (np.nan_to_num(p_values[name], nan=1.0), name))
        worst_p_value = np.nan_to_num(p_values[worst_term], nan=1.0)
        if worst_p_value <= alpha:
            return active_terms, last_p_values
        active_terms.remove(worst_term)

    raise RuntimeError(f'No MLG predictor is significant at alpha={alpha:.3f}.')


def _fit_regularized_multinomial(design, y, regularization_c=DEFAULT_BOOTSTRAP_C, random_state=42):
    if design.shape[1] == 0:
        return _PriorMultinomialModel(y)

    model = LogisticRegression(
        C=float(regularization_c),
        penalty='l2',
        solver='lbfgs',
        max_iter=4000,
        random_state=random_state,
    )
    model.fit(design, y)
    return model


class _PriorMultinomialModel:
    """Intercept-only multinomial model used by bootstrap reduced designs."""

    def __init__(self, y):
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = values
        self.probabilities_ = counts / counts.sum()

    def predict_proba(self, design):
        return np.tile(self.probabilities_, (design.shape[0], 1))


def _aligned_model_probabilities(model, design, classes):
    probabilities = np.zeros((design.shape[0], len(classes)))
    predicted = model.predict_proba(design)
    class_to_column = {int(state): index for index, state in enumerate(classes)}
    for model_column, state in enumerate(model.classes_):
        probabilities[:, class_to_column[int(state)]] = predicted[:, model_column]
    return np.clip(probabilities, 1e-12, 1.0)


def _multinomial_log_likelihood(y, probabilities, classes):
    class_to_column = {int(state): index for index, state in enumerate(classes)}
    columns = [class_to_column[int(state)] for state in y]
    return float(np.log(probabilities[np.arange(len(y)), columns]).sum())


def _regularized_term_improvement(
    transformer,
    X,
    y,
    active_terms,
    removed_term,
    classes,
    regularization_c,
    random_state,
):
    reduced_terms = [term for term in active_terms if term != removed_term]
    full_design = transformer.transform(X, active_terms)
    reduced_design = transformer.transform(X, reduced_terms)
    full_model = _fit_regularized_multinomial(
        full_design, y, regularization_c=regularization_c, random_state=random_state
    )
    reduced_model = _fit_regularized_multinomial(
        reduced_design, y, regularization_c=regularization_c, random_state=random_state
    )
    full_probabilities = _aligned_model_probabilities(full_model, full_design, classes)
    reduced_probabilities = _aligned_model_probabilities(reduced_model, reduced_design, classes)
    return (
        _multinomial_log_likelihood(y, full_probabilities, classes)
        - _multinomial_log_likelihood(y, reduced_probabilities, classes)
    )


def _bootstrap_term_p_value(
    transformer,
    X_train,
    y_train,
    active_terms,
    term_name,
    regularization_c,
    n_bootstrap,
    alpha,
    rng,
):
    """Parametric bootstrap LR-style test under the model without one term."""
    classes = np.array(sorted(pd.unique(y_train)))
    observed = _regularized_term_improvement(
        transformer,
        X_train,
        y_train,
        active_terms,
        term_name,
        classes,
        regularization_c,
        random_state=BOOTSTRAP_RANDOM_STATE,
    )

    reduced_terms = [term for term in active_terms if term != term_name]
    reduced_design = transformer.transform(X_train, reduced_terms)
    reduced_model = _fit_regularized_multinomial(
        reduced_design,
        y_train,
        regularization_c=regularization_c,
        random_state=BOOTSTRAP_RANDOM_STATE,
    )
    null_probabilities = _aligned_model_probabilities(reduced_model, reduced_design, classes)
    extreme_count = 0
    valid_count = 0
    failure_cutoff = max(0, int(np.floor(alpha_count_limit(n_bootstrap, alpha))))

    for replicate in range(int(n_bootstrap)):
        simulated = np.array([
            rng.choice(classes, p=row / row.sum())
            for row in null_probabilities
        ])
        if np.unique(simulated).size < len(classes):
            continue
        simulated_y = pd.Series(simulated, index=y_train.index)
        try:
            improvement = _regularized_term_improvement(
                transformer,
                X_train,
                simulated_y,
                active_terms,
                term_name,
                classes,
                regularization_c,
                random_state=BOOTSTRAP_RANDOM_STATE + replicate + 1,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        valid_count += 1
        if improvement >= observed:
            extreme_count += 1
            if extreme_count > failure_cutoff:
                break

    if valid_count == 0:
        raise RuntimeError(f'Bootstrap p-value for {term_name} had no valid replicates.')
    p_value = (extreme_count + 1.0) / (valid_count + 1.0)
    return p_value, observed, valid_count


def alpha_count_limit(n_bootstrap, alpha=SIGNIFICANCE_LEVEL):
    """Largest extreme bootstrap count that can still pass the empirical alpha cutoff."""
    return alpha * (int(n_bootstrap) + 1) - 1


def select_significant_terms_bootstrap(
    transformer,
    X_train,
    y_train,
    alpha=SIGNIFICANCE_LEVEL,
    regularization_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Screen whole terms against an intercept-only baseline with regularized bootstrap tests."""
    rng = np.random.default_rng(random_state)
    p_values = {}
    improvements = {}
    replicates = {}

    for term_name in transformer.term_names_:
        p_value, improvement, valid_count = _bootstrap_term_p_value(
            transformer,
            X_train,
            y_train,
            [term_name],
            term_name,
            regularization_c,
            n_bootstrap,
            alpha,
            rng,
        )
        p_values[term_name] = p_value
        improvements[term_name] = improvement
        replicates[term_name] = valid_count

    selected_terms = [
        term for term in transformer.term_names_
        if np.nan_to_num(p_values[term], nan=1.0) <= alpha
    ]
    if selected_terms:
        return selected_terms, p_values, improvements, replicates
    raise RuntimeError(f'No MLG predictor is significant at alpha={alpha:.3f}.')


def select_significant_terms(
    transformer,
    X_train,
    y_train,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Select terms with Wald tests, or regularized bootstrap tests when requested/needed."""
    if inference_method not in {'auto', 'wald', 'regularized_bootstrap'}:
        raise ValueError(f'Unknown MLG inference method: {inference_method}')

    if inference_method in {'auto', 'wald'}:
        try:
            selected, p_values = select_significant_terms_wald(
                transformer, X_train, y_train, alpha=alpha
            )
            return selected, p_values, 'wald', {}, {}
        except RuntimeError:
            if inference_method == 'wald':
                raise

    selected, p_values, improvements, replicates = select_significant_terms_bootstrap(
        transformer,
        X_train,
        y_train,
        alpha=alpha,
        regularization_c=bootstrap_c,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return selected, p_values, 'regularized_bootstrap', improvements, replicates


@dataclass
class PreparedMLGDesign:
    transformer: MLGFeatureTransformer
    selected_terms: list
    term_p_values: dict
    X_train: pd.DataFrame
    y_train: pd.Series
    inference_method: str = 'wald'
    term_improvements: dict = None
    bootstrap_replicates: dict = None


class MLGTransitionModel:
    def __init__(self, prepared_design, regularization_c, random_state=42):
        self.transformer = prepared_design.transformer
        self.selected_terms = list(prepared_design.selected_terms)
        self.term_p_values = dict(prepared_design.term_p_values)
        self.inference_method = prepared_design.inference_method
        self.regularization_c = float(regularization_c)
        self.classifier = LogisticRegression(
            C=self.regularization_c,
            penalty='l2',
            solver='lbfgs',
            max_iter=4000,
            random_state=random_state,
        )
        design = self.transformer.transform(prepared_design.X_train, self.selected_terms)
        self.classifier.fit(design, prepared_design.y_train)

    @property
    def classes_(self):
        return self.classifier.classes_

    def predict_proba(self, X):
        design = self.transformer.transform(X, self.selected_terms)
        return self.classifier.predict_proba(design)

    def term_log_odds(self, X, term_name):
        """Return one term's contribution to relevant class log-odds."""
        if term_name not in self.selected_terms:
            raise ValueError(f'{term_name} is not in the selected MLG model.')

        term_matrix = self.transformer.transform_term(X, term_name)
        slices = self.transformer.term_slices(X, self.selected_terms)
        term_slice = slices[term_name]
        coefficients = self.classifier.coef_[:, term_slice]

        if len(self.classes_) == 2:
            effects = term_matrix @ coefficients[0]
            labels = [f'{self.classes_[1]} vs {self.classes_[0]}']
            return effects.reshape(-1, 1), labels

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

        pairwise_effects = []
        labels = []
        for reference_index in range(len(self.classes_) - 1):
            for contrast_index in range(reference_index + 1, len(self.classes_)):
                pairwise_effects.append(
                    term_matrix @ (coefficients[contrast_index] - coefficients[reference_index])
                )
                labels.append(
                    f'{self.classes_[contrast_index]} vs {self.classes_[reference_index]}'
                )
        effects = np.column_stack(pairwise_effects)
        return effects, labels


def prepare_mlg_design(
    X_train,
    y_train,
    n_knots,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    transformer = MLGFeatureTransformer(n_knots=n_knots).fit(X_train)
    selected_terms, p_values, method, improvements, replicates = select_significant_terms(
        transformer,
        X_train,
        y_train,
        alpha=alpha,
        inference_method=inference_method,
        bootstrap_c=bootstrap_c,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return PreparedMLGDesign(
        transformer,
        selected_terms,
        p_values,
        X_train,
        y_train,
        method,
        improvements,
        replicates,
    )


def fit_mlg_transition_model(
    X_train,
    y_train,
    n_knots,
    regularization_c,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    prepared = prepare_mlg_design(
        X_train,
        y_train,
        n_knots,
        alpha,
        inference_method=inference_method,
        bootstrap_c=bootstrap_c,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return MLGTransitionModel(prepared, regularization_c)


def _read_transition_inputs(n_states):
    stock_rets = pd.read_csv(
        BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv',
        index_col=[0],
        parse_dates=[0],
    )
    exog_rets = pd.read_csv(
        BASE_DIR / 'processed' / 'exog_rets.csv',
        index_col=[0],
        parse_dates=[0],
    )
    return stock_rets['STATE'], exog_rets


def run_mlg_grid(
    n_states,
    state_lags=(0, 1, 2),
    exog_lags=(0, 1, 2),
    knot_grid=DEFAULT_KNOTS,
    regularization_grid=DEFAULT_REGULARIZATION_C,
    alpha=SIGNIFICANCE_LEVEL,
    inference_method='auto',
    bootstrap_c=DEFAULT_BOOTSTRAP_C,
    n_bootstrap=DEFAULT_BOOTSTRAP_REPLICATES,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    states, exog_rets = _read_transition_inputs(n_states)
    rows = []
    errors = []

    for state_lag in state_lags:
        for exog_lag in exog_lags:
            X, y, meta, state_classes = build_transition_dataset(
                states,
                exog_rets,
                max_state_lag=state_lag,
                max_exog_lag=exog_lag,
            )
            splits = split_transition_dataset(X, y, meta)
            X_train, y_train, _ = splits['train']

            for n_knots in knot_grid:
                try:
                    prepared = prepare_mlg_design(
                        X_train,
                        y_train,
                        n_knots,
                        alpha,
                        inference_method=inference_method,
                        bootstrap_c=bootstrap_c,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    errors.append({
                        'N_STATES': n_states,
                        'STATE_LAG': state_lag,
                        'EXOG_LAG': exog_lag,
                        'N_KNOTS': n_knots,
                        'ERROR': str(error),
                    })
                    continue

                selected_p_values = {
                    term: prepared.term_p_values[term]
                    for term in prepared.selected_terms
                }
                common = {
                    'N_STATES': n_states,
                    'STATE_LAG': state_lag,
                    'EXOG_LAG': exog_lag,
                    'N_KNOTS': n_knots,
                    'SIGNIFICANCE_LEVEL': alpha,
                    'SELECTED_TERM_COUNT': len(prepared.selected_terms),
                    'SELECTED_TERMS': '|'.join(prepared.selected_terms),
                    'MAX_SELECTED_P_VALUE': max(selected_p_values.values()),
                    'ALL_TERMS_SIGNIFICANT': all(value <= alpha for value in selected_p_values.values()),
                    'TERM_P_VALUES': json.dumps(prepared.term_p_values, sort_keys=True),
                    'INFERENCE_METHOD': prepared.inference_method,
                    'BOOTSTRAP_C': float(bootstrap_c),
                    'BOOTSTRAP_REPLICATES': int(n_bootstrap) if prepared.inference_method == 'regularized_bootstrap' else 0,
                    'TERM_BOOTSTRAP_IMPROVEMENTS': json.dumps(prepared.term_improvements or {}, sort_keys=True),
                    'TERM_BOOTSTRAP_VALID_REPLICATES': json.dumps(prepared.bootstrap_replicates or {}, sort_keys=True),
                    'TRAIN_ROWS': len(X_train),
                }

                for regularization_c in regularization_grid:
                    model = MLGTransitionModel(prepared, regularization_c)
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

    error_path = BASE_DIR / 'phase_1' / f'mlg_transition_errors_q{n_states}.csv'
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    elif error_path.exists():
        # Do not retain an error report from a failed run after a clean rerun.
        error_path.unlink()
    if not rows:
        raise RuntimeError(
            f'Every MLG candidate failed for q{n_states}. '
            f'See phase_1/mlg_transition_errors_q{n_states}.csv.'
        )
    return pd.DataFrame(rows)


def best_mlg_config(report):
    significant = report['ALL_TERMS_SIGNIFICANT']
    if significant.dtype != bool:
        significant = significant.astype(str).str.lower().eq('true')
    candidates = report[
        (report['MODEL'] == MODEL_NAME)
        & (report['SPLIT'] == 'validation')
        & significant
    ].copy()
    if candidates.empty:
        raise ValueError('No significant MLG validation candidate is available.')
    return candidates.sort_values(
        ['LOG_LOSS', 'BRIER_SCORE', 'ACCURACY', 'SELECTED_TERM_COUNT',
         'STATE_LAG', 'EXOG_LAG', 'N_KNOTS', 'REGULARIZATION_C'],
        ascending=[True, True, False, True, True, True, True, True],
    ).iloc[0]


def save_mlg_transition_report(n_states, **grid_options):
    report = run_mlg_grid(n_states=n_states, **grid_options)
    output_path = BASE_DIR / 'phase_1' / f'mlg_transition_report_q{n_states}.csv'
    report.to_csv(output_path, index=False)

    best = best_mlg_config(report)
    p_values = json.loads(best['TERM_P_VALUES'])
    selected_terms = set(str(best['SELECTED_TERMS']).split('|'))
    significance = pd.DataFrame([
        {
            'N_STATES': n_states,
            'TERM': term,
            'P_VALUE': p_value,
            'SELECTED': term in selected_terms,
            'SIGNIFICANT_AT_5_PERCENT': p_value <= SIGNIFICANCE_LEVEL,
        }
        for term, p_value in p_values.items()
    ])
    significance.to_csv(
        BASE_DIR / 'phase_1' / f'mlg_selected_significance_q{n_states}.csv',
        index=False,
    )
    return report, output_path


def load_or_create_mlg_report(n_states):
    path = BASE_DIR / 'phase_1' / f'mlg_transition_report_q{n_states}.csv'
    if path.exists():
        report = pd.read_csv(path)
        required = {'N_KNOTS', 'REGULARIZATION_C', 'TERM_P_VALUES'}
        if required.issubset(report.columns):
            return report
    report, _ = save_mlg_transition_report(n_states)
    return report


def fit_selected_mlg_transition(n_states, report=None):
    report = load_or_create_mlg_report(n_states) if report is None else report
    best = best_mlg_config(report)
    states, exog_rets = _read_transition_inputs(n_states)
    X, y, meta, state_classes = build_transition_dataset(
        states,
        exog_rets,
        max_state_lag=int(best['STATE_LAG']),
        max_exog_lag=int(best['EXOG_LAG']),
    )
    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']
    model = fit_mlg_transition_model(
        X_train,
        y_train,
        n_knots=int(best['N_KNOTS']),
        regularization_c=float(best['REGULARIZATION_C']),
        alpha=float(best['SIGNIFICANCE_LEVEL']),
        inference_method=str(best.get('INFERENCE_METHOD', 'auto')),
        bootstrap_c=float(best.get('BOOTSTRAP_C', DEFAULT_BOOTSTRAP_C)),
        n_bootstrap=int(best.get('BOOTSTRAP_REPLICATES', DEFAULT_BOOTSTRAP_REPLICATES)) or DEFAULT_BOOTSTRAP_REPLICATES,
    )
    return model, splits, state_classes, best


def _partial_effect_frame(model, X_train, term_name):
    term = model.transformer.term_spec(term_name)
    reference = X_train.median(axis=0).to_frame().T

    if term.kind == 'categorical':
        labels = [column.rsplit('_', 1)[-1] for column in term.columns]
        frame = pd.concat([reference] * len(labels), ignore_index=True)
        frame.loc[:, term.columns] = 0.0
        for row, column in enumerate(term.columns):
            frame.loc[row, column] = 1.0
        values = np.arange(len(labels), dtype=float)
        center_index = len(labels) - 1
    else:
        observed = X_train[term.columns[0]].to_numpy(dtype=float)
        unique_values = np.unique(observed)
        if unique_values.size <= 80:
            values = unique_values
        else:
            lower, upper = np.quantile(observed, [0.01, 0.99])
            values = np.linspace(lower, upper, 80)
        labels = [f'{value:.6g}' for value in values]
        frame = pd.concat([reference] * len(values), ignore_index=True)
        frame.loc[:, term.columns[0]] = values
        center_index = int(np.abs(values - np.median(observed)).argmin())

    effects, contrasts = model.term_log_odds(frame, term_name)
    effects = effects - effects[center_index]
    return values, labels, effects, contrasts, term.kind


def plot_selected_mlg_partial_effects(n_states, model, X_train, output_dir):
    """Save one centered log-odds partial-effect plot per selected predictor."""
    output_dir = Path(output_dir) / f'mlg_partial_effects_q{n_states}'
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    paths = []

    for term_name in model.selected_terms:
        values, labels, effects, contrasts, kind = _partial_effect_frame(model, X_train, term_name)
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for column, contrast in enumerate(contrasts):
            if kind == 'categorical':
                ax.plot(values, effects[:, column], marker='o', label=contrast)
            else:
                ax.plot(values, effects[:, column], linewidth=2, label=contrast)
            for row, value in enumerate(values):
                records.append({
                    'N_STATES': n_states,
                    'TERM': term_name,
                    'P_VALUE': model.term_p_values[term_name],
                    'PREDICTOR_VALUE': value,
                    'PREDICTOR_LEVEL': labels[row] if kind == 'categorical' else None,
                    'LOG_ODDS_CONTRAST': contrast,
                    'CENTERED_PARTIAL_EFFECT': effects[row, column],
                })

        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_title(f'q{n_states} MLG partial effect: {term_name}')
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
