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
from scipy.stats import beta as beta_distribution
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


def _contrast_states(contrast):
    """Return the numerator and denominator state labels from ``a vs b``."""
    match = re.fullmatch(r'(-?\d+) vs (-?\d+)', contrast)
    if match is None:
        raise ValueError(f'Unsupported log-odds contrast: {contrast}')
    return int(match.group(1)), int(match.group(2))


def _duration_focal_state(term_name):
    match = re.fullmatch(r'DURATION_STATE_(\d+)', term_name)
    return int(match.group(1)) if match else None


def _term_values(X, term):
    """Return a scalar plotting value for a raw model term."""
    if term.kind == 'categorical':
        return np.argmax(X[term.columns].to_numpy(dtype=float), axis=1).astype(float)
    return X[term.columns[0]].to_numpy(dtype=float)


def _term_grid(X, term):
    values = _term_values(X, term)
    unique_values = np.unique(values)
    if term.kind == 'categorical' or unique_values.size <= 80:
        return unique_values
    return np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 80)


def _set_term_value(frame, term, value):
    """Set one raw term consistently before calculating partial dependence."""
    if term.kind == 'categorical':
        frame.loc[:, term.columns] = 0.0
        frame.loc[:, term.columns[int(value)]] = 1.0
    else:
        frame.loc[:, term.columns[0]] = value


def _relevant_term_sample(X_train, y_train, meta_train, term_name):
    """Use the current state matching a state-specific duration term."""
    sample = X_train.copy()
    sample['_NEXT_STATE'] = pd.Series(y_train, index=X_train.index)
    focal_state = _duration_focal_state(term_name)
    if focal_state is not None:
        if meta_train is None or 'CURRENT_STATE' not in meta_train:
            raise ValueError(
                'meta_train with CURRENT_STATE is required for state-duration plots.'
            )
        current_state = meta_train['CURRENT_STATE'].reindex(X_train.index)
        sample = sample.loc[current_state == focal_state].copy()
    return sample


def _binned_empirical_log_odds(values, next_states, numerator, denominator, categorical):
    """Estimate observed pairwise log-odds with binning and Jeffreys smoothing."""
    frame = pd.DataFrame({
        'VALUE': values,
        'NEXT_STATE': next_states,
    }).dropna()
    frame = frame[frame['NEXT_STATE'].isin([numerator, denominator])].copy()
    if frame.empty:
        return pd.DataFrame()

    unique_values = frame['VALUE'].nunique()
    if categorical or unique_values <= 12:
        frame['_BIN'] = frame['VALUE']
    else:
        # Quantile bins retain observations in sparse tails without hiding the trend.
        n_bins = min(10, max(4, len(frame) // 25))
        try:
            frame['_BIN'] = pd.qcut(
                frame['VALUE'], q=n_bins, labels=False, duplicates='drop'
            )
        except ValueError:
            frame['_BIN'] = 0

    records = []
    for _, group in frame.groupby('_BIN', observed=True):
        numerator_count = int((group['NEXT_STATE'] == numerator).sum())
        denominator_count = int((group['NEXT_STATE'] == denominator).sum())
        numerator_smoothed = numerator_count + 0.5
        denominator_smoothed = denominator_count + 0.5
        log_odds = np.log(numerator_smoothed / denominator_smoothed)
        records.append({
            'PREDICTOR_VALUE': float(group['VALUE'].median()),
            'OBSERVATION_COUNT': int(len(group)),
            'NUMERATOR_COUNT': numerator_count,
            'DENOMINATOR_COUNT': denominator_count,
            'OBSERVED_LOG_ODDS': log_odds,
            'OBSERVED_LOG_ODDS_SE': np.sqrt(
                1.0 / numerator_smoothed + 1.0 / denominator_smoothed
            ),
        })
    return pd.DataFrame(records).sort_values('PREDICTOR_VALUE').reset_index(drop=True)


def _fitted_pairwise_log_odds(model, sample_X, term, values, numerator, denominator):
    """Average fitted pairwise log-odds after varying only the plotted term."""
    classes = np.asarray(model.classes_)
    numerator_index = int(np.where(classes == numerator)[0][0])
    denominator_index = int(np.where(classes == denominator)[0][0])
    fitted = []
    for value in values:
        counterfactual = sample_X.copy()
        _set_term_value(counterfactual, term, value)
        probabilities = model.predict_proba(counterfactual)
        fitted.append(float(np.mean(np.log(
            np.clip(probabilities[:, numerator_index], 1e-12, 1.0)
            / np.clip(probabilities[:, denominator_index], 1e-12, 1.0)
        ))))
    return np.asarray(fitted)


def _common_duration_bins(values, next_states, state_classes, max_bins=10):
    """Create common duration bins and retain counts for every next state."""
    frame = pd.DataFrame({
        'VALUE': values,
        'NEXT_STATE': next_states,
    }).dropna()
    if frame.empty:
        return pd.DataFrame()

    unique_values = frame['VALUE'].nunique()
    if unique_values <= 12:
        frame['_BIN'] = frame['VALUE']
    else:
        n_bins = min(max_bins, max(4, len(frame) // 25))
        try:
            frame['_BIN'] = pd.qcut(
                frame['VALUE'], q=n_bins, labels=False, duplicates='drop'
            )
        except ValueError:
            frame['_BIN'] = 0

    records = []
    for bin_index, (_, group) in enumerate(frame.groupby('_BIN', observed=True)):
        record = {
            'BIN_INDEX': bin_index,
            'BIN_LOWER': float(group['VALUE'].min()),
            'BIN_MEDIAN': float(group['VALUE'].median()),
            'BIN_UPPER': float(group['VALUE'].max()),
            'OBSERVATION_COUNT': int(len(group)),
        }
        for state in state_classes:
            record[f'NEXT_STATE_{state}_COUNT'] = int(
                (group['NEXT_STATE'] == state).sum()
            )
        records.append(record)
    return pd.DataFrame(records)


def _jeffreys_log_odds_interval(numerator_count, denominator_count, alpha=0.05):
    """Return a pairwise log-odds estimate and Jeffreys credible interval."""
    numerator_shape = float(numerator_count) + 0.5
    denominator_shape = float(denominator_count) + 0.5
    estimate = np.log(numerator_shape / denominator_shape)
    probability_bounds = beta_distribution.ppf(
        [alpha / 2, 1 - alpha / 2], numerator_shape, denominator_shape
    )
    probability_bounds = np.clip(probability_bounds, 1e-12, 1 - 1e-12)
    log_odds_bounds = np.log(probability_bounds / (1 - probability_bounds))
    return estimate, float(log_odds_bounds[0]), float(log_odds_bounds[1])


def _duration_bin_label(row):
    lower = row['BIN_LOWER']
    upper = row['BIN_UPPER']
    if lower == upper:
        return f'{lower:g}'
    return f'{lower:g}-{upper:g}'


def _plot_combined_duration_contrasts(
    n_states,
    model,
    sample_X,
    next_states,
    term,
    term_name,
    contrasts,
    grid,
    output_dir,
    model_label,
    extra_record_fields,
):
    """Combine all stay-versus-transition duration contrasts on common bins."""
    focal_state = _duration_focal_state(term_name)
    if focal_state is None or len(contrasts) <= 1:
        return None, []

    state_classes = [int(state) for state in model.classes_]
    bins = _common_duration_bins(
        _term_values(sample_X, term), next_states, state_classes
    )
    if bins.empty:
        return None, []

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    markers = ['o', 's', '^', 'D', 'v']
    line_styles = ['-', '--', '-.', ':']
    fig, (ax, count_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.6),
        gridspec_kw={'height_ratios': [3.2, 1.35]},
        constrained_layout=True,
    )
    records = []
    x_span = max(float(grid.max() - grid.min()), 1.0)
    offsets = np.linspace(-0.0075 * x_span, 0.0075 * x_span, len(contrasts))

    for contrast_index, contrast in enumerate(contrasts):
        numerator, denominator = _contrast_states(contrast)
        fitted = _fitted_pairwise_log_odds(
            model, sample_X, term, grid, numerator, denominator
        )
        color = colors[contrast_index % len(colors)]
        ax.plot(
            grid,
            fitted,
            color=color,
            linestyle=line_styles[contrast_index % len(line_styles)],
            linewidth=2.2,
            label=contrast,
        )

        observed_x = bins['BIN_MEDIAN'].to_numpy(dtype=float) + offsets[contrast_index]
        observed_log_odds = []
        lower_bounds = []
        upper_bounds = []
        for row in bins.to_dict('records'):
            numerator_count = row[f'NEXT_STATE_{numerator}_COUNT']
            denominator_count = row[f'NEXT_STATE_{denominator}_COUNT']
            estimate, lower, upper = _jeffreys_log_odds_interval(
                numerator_count, denominator_count
            )
            observed_log_odds.append(estimate)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            record = {
                'N_STATES': n_states,
                'TERM': term_name,
                'P_VALUE': model.term_p_values[term_name],
                'ROW_TYPE': 'OBSERVED_BIN',
                'LOG_ODDS_CONTRAST': contrast,
                'BIN_INDEX': row['BIN_INDEX'],
                'BIN_LOWER': row['BIN_LOWER'],
                'BIN_MEDIAN': row['BIN_MEDIAN'],
                'BIN_UPPER': row['BIN_UPPER'],
                'OBSERVATION_COUNT': row['OBSERVATION_COUNT'],
                'NUMERATOR_COUNT': numerator_count,
                'DENOMINATOR_COUNT': denominator_count,
                'OBSERVED_LOG_ODDS': estimate,
                'CI_LOWER': lower,
                'CI_UPPER': upper,
            }
            for state in state_classes:
                record[f'NEXT_STATE_{state}_COUNT'] = row[f'NEXT_STATE_{state}_COUNT']
            record.update(extra_record_fields.get(term_name, {}))
            records.append(record)

        observed_log_odds = np.asarray(observed_log_odds)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        ax.errorbar(
            observed_x,
            observed_log_odds,
            yerr=np.vstack([
                observed_log_odds - lower_bounds,
                upper_bounds - observed_log_odds,
            ]),
            fmt=markers[contrast_index % len(markers)],
            markersize=5.5,
            markerfacecolor='white',
            markeredgewidth=1.5,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
        )
        for value, fitted_value in zip(grid, fitted):
            record = {
                'N_STATES': n_states,
                'TERM': term_name,
                'P_VALUE': model.term_p_values[term_name],
                'ROW_TYPE': 'FITTED_CURVE',
                'LOG_ODDS_CONTRAST': contrast,
                'PREDICTOR_VALUE': value,
                'FITTED_LOG_ODDS': fitted_value,
            }
            record.update(extra_record_fields.get(term_name, {}))
            records.append(record)

    ax.axhline(0, color='black', linewidth=0.9, linestyle=':')
    ax.set_title(
        f'q{n_states} {model_label}: duration in state {focal_state}', pad=10
    )
    ax.set_xlabel(f'Days currently in state {focal_state}')
    ax.set_ylabel(f'log[P(next={focal_state}) / P(next=other state)]')
    ax.legend(title='Next-state contrast', ncol=min(3, len(contrasts)), loc='best')
    ax.text(
        0.01,
        0.02,
        'Lines: fitted MLG; open markers: observed common-bin log-odds; bars: 95% Jeffreys intervals',
        transform=ax.transAxes,
        fontsize=8,
        va='bottom',
    )

    count_matrix = bins[
        [f'NEXT_STATE_{state}_COUNT' for state in state_classes]
    ].to_numpy(dtype=float).T
    count_image = count_ax.imshow(
        np.log1p(count_matrix), aspect='auto', cmap='Blues', interpolation='nearest'
    )
    max_intensity = max(float(np.log1p(count_matrix).max()), 1.0)
    for row_index in range(count_matrix.shape[0]):
        for column_index in range(count_matrix.shape[1]):
            intensity = np.log1p(count_matrix[row_index, column_index]) / max_intensity
            count_ax.text(
                column_index,
                row_index,
                f'{int(count_matrix[row_index, column_index])}',
                ha='center',
                va='center',
                color='white' if intensity > 0.58 else 'black',
                fontsize=8,
            )
    count_ax.set_yticks(np.arange(len(state_classes)))
    count_ax.set_yticklabels([f'Next state {state}' for state in state_classes])
    count_ax.set_xticks(np.arange(len(bins)))
    count_ax.set_xticklabels(
        [_duration_bin_label(row) for row in bins.to_dict('records')], rotation=30
    )
    count_ax.set_xlabel(f'Duration-in-state-{focal_state} bin (days)')
    count_ax.set_title('Observed transition counts in each common duration bin', fontsize=10)
    count_image.set_clim(0, max_intensity)

    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', term_name)
    path = output_dir / f'{safe_name}__all_contrasts.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path, records


def _plot_selected_transition_effects(
    n_states,
    model,
    X_train,
    y_train,
    meta_train,
    output_dir,
    directory_name,
    model_label,
    extra_record_fields=None,
):
    """Plot one observed-versus-fitted log-odds chart for every term contrast."""
    output_dir = Path(output_dir) / directory_name
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    combined_duration_records = []
    paths = []
    extra_record_fields = extra_record_fields or {}

    for term_name in model.selected_terms:
        term = model.transformer.term_spec(term_name)
        sample = _relevant_term_sample(X_train, y_train, meta_train, term_name)
        if sample.empty:
            continue
        sample_X = sample[X_train.columns]
        term_values = _term_values(sample_X, term)
        grid = _term_grid(sample_X, term)
        _, labels, _, contrasts, _ = _partial_effect_frame(model, sample_X, term_name)

        combined_path, combined_records = _plot_combined_duration_contrasts(
            n_states,
            model,
            sample_X,
            sample['_NEXT_STATE'].to_numpy(dtype=int),
            term,
            term_name,
            contrasts,
            grid,
            output_dir,
            model_label,
            extra_record_fields,
        )
        if combined_path is not None:
            paths.append(combined_path)
            combined_duration_records.extend(combined_records)

        for contrast in contrasts:
            numerator, denominator = _contrast_states(contrast)
            observed = _binned_empirical_log_odds(
                term_values,
                sample['_NEXT_STATE'].to_numpy(dtype=int),
                numerator,
                denominator,
                categorical=term.kind == 'categorical',
            )
            fitted = _fitted_pairwise_log_odds(
                model, sample_X, term, grid, numerator, denominator
            )

            fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
            ax.plot(
                grid, fitted, color='#1f5a85', linewidth=2.2,
                label='Fitted MLG partial-dependence curve',
            )
            if not observed.empty:
                ax.errorbar(
                    observed['PREDICTOR_VALUE'], observed['OBSERVED_LOG_ODDS'],
                    yerr=1.96 * observed['OBSERVED_LOG_ODDS_SE'],
                    fmt='o', color='#b04a24', ecolor='#b04a24', capsize=3,
                    label='Observed binned log-odds (95% approximate CI)',
                )
            ax.axhline(0, color='black', linewidth=1, linestyle='--')
            ax.set_title(f'q{n_states} {model_label}: {term_name} | {contrast}')
            ax.set_xlabel(term_name)
            ax.set_ylabel(f'log[P(next={numerator}) / P(next={denominator})]')
            if term.kind == 'categorical':
                ax.set_xticks(grid)
                ax.set_xticklabels(labels)
            ax.legend(loc='best', fontsize=8)

            safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', term_name)
            contrast_name = f'{numerator}_vs_{denominator}'
            path = output_dir / f'{safe_name}__{contrast_name}.png'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            plt.close(fig)
            paths.append(path)

            for value, fitted_value in zip(grid, fitted):
                record = {
                    'N_STATES': n_states,
                    'TERM': term_name,
                    'P_VALUE': model.term_p_values[term_name],
                    'PREDICTOR_VALUE': value,
                    'LOG_ODDS_CONTRAST': contrast,
                    'FITTED_LOG_ODDS': fitted_value,
                }
                record.update(extra_record_fields.get(term_name, {}))
                records.append(record)
            for row in observed.to_dict('records'):
                record = {
                    'N_STATES': n_states,
                    'TERM': term_name,
                    'P_VALUE': model.term_p_values[term_name],
                    'PREDICTOR_VALUE': row['PREDICTOR_VALUE'],
                    'LOG_ODDS_CONTRAST': contrast,
                    'OBSERVATION_COUNT': row['OBSERVATION_COUNT'],
                    'NUMERATOR_COUNT': row['NUMERATOR_COUNT'],
                    'DENOMINATOR_COUNT': row['DENOMINATOR_COUNT'],
                    'OBSERVED_LOG_ODDS': row['OBSERVED_LOG_ODDS'],
                    'OBSERVED_LOG_ODDS_SE': row['OBSERVED_LOG_ODDS_SE'],
                }
                record.update(extra_record_fields.get(term_name, {}))
                records.append(record)

    effects_path = output_dir / 'partial_effects.csv'
    pd.DataFrame(records).to_csv(effects_path, index=False)
    if combined_duration_records:
        pd.DataFrame(combined_duration_records).to_csv(
            output_dir / 'duration_all_contrasts.csv', index=False
        )
    return paths, effects_path


def plot_selected_mlg_partial_effects(
    n_states, model, X_train, output_dir, y_train=None, meta_train=None
):
    """Save readable observed-versus-fitted charts for every selected MLG contrast."""
    if y_train is None:
        raise ValueError('y_train is required to plot observed transition effects.')
    if meta_train is None:
        raise ValueError('meta_train is required to plot observed transition effects.')
    return _plot_selected_transition_effects(
        n_states,
        model,
        X_train,
        y_train,
        meta_train,
        output_dir,
        f'mlg_partial_effects_q{n_states}',
        'MLG observed vs fitted transition effect',
    )
