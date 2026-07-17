"""Goodness-of-fit, regime-count, and stability diagnostics for Gaussian HMMs."""

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import chi2, chi2_contingency, jarque_bera, kstest, multivariate_normal, norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from p1_4_dynamic_transition import TRAIN_END, VALIDATION_END
from project_config import BASE_DIR


DIAGNOSTIC_STATES = (2, 3, 4)
DIAGNOSTIC_FEATURES = (
    'SPXT',
    'DBLCIX',
    'IBOXIG',
    'JPEICORE',
    'LT11TRUU',
    'LBUTTRUU',
)
SIGNIFICANCE_LEVEL = 0.05
DEFAULT_N_STARTS = 50
DEFAULT_BOOTSTRAP_REPLICATES = 100
DEFAULT_BOOTSTRAP_STARTS = 3
DEFAULT_RANDOM_SEED = 42
DEFAULT_COVARIANCE_CONDITION_THRESHOLD = 1e8
DEFAULT_DURATION_BOOTSTRAP_REPLICATES = 500


def _fit_hmm(data, n_states, seed, max_iter=2000):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        random_state=int(seed),
        n_iter=max_iter,
        tol=1e-5,
        min_covar=1e-6,
        implementation='scaling',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(data)
    return model


def fit_multistart_hmms(
    train_data,
    state_counts=DIAGNOSTIC_STATES,
    n_starts=DEFAULT_N_STARTS,
    random_seed=DEFAULT_RANDOM_SEED,
    min_regime_count=10,
    max_iter=2000,
):
    """Fit each regime count from identical starts and retain its best valid likelihood."""
    values = np.asarray(train_data, dtype=float)
    seeds = np.random.default_rng(random_seed).choice(
        2 ** 31 - 1,
        size=int(n_starts),
        replace=False,
    )
    best_models = {}
    candidate_rows = []

    for n_states in state_counts:
        fitted = []
        for seed in seeds:
            try:
                model = _fit_hmm(values, n_states, seed, max_iter=max_iter)
                states = model.predict(values)
                counts = np.bincount(states, minlength=n_states)
                log_likelihood = float(model.score(values))
                row = {
                    'N_STATES': n_states,
                    'SEED': int(seed),
                    'TRAIN_LOG_LIKELIHOOD': log_likelihood,
                    'AIC': float(model.aic(values)),
                    'BIC': float(model.bic(values)),
                    'CONVERGED': bool(model.monitor_.converged),
                    'ALL_STATES_USED': bool(np.all(counts > 0)),
                    'MIN_REGIME_COUNT': int(counts.min()),
                    'VALID_FOR_SELECTION': bool(
                        model.monitor_.converged and np.all(counts >= min_regime_count)
                    ),
                    'ERROR': None,
                }
                candidate_rows.append(row)
                fitted.append((model, row))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                candidate_rows.append({
                    'N_STATES': n_states,
                    'SEED': int(seed),
                    'TRAIN_LOG_LIKELIHOOD': np.nan,
                    'AIC': np.nan,
                    'BIC': np.nan,
                    'CONVERGED': False,
                    'ALL_STATES_USED': False,
                    'MIN_REGIME_COUNT': 0,
                    'VALID_FOR_SELECTION': False,
                    'ERROR': str(error),
                })

        valid = [item for item in fitted if item[1]['VALID_FOR_SELECTION']]
        if not valid:
            valid = [
                item for item in fitted
                if item[1]['CONVERGED'] and item[1]['ALL_STATES_USED']
            ]
        if not valid:
            raise RuntimeError(f'No converged q{n_states} diagnostic HMM used every state.')
        best_models[n_states] = max(
            valid,
            key=lambda item: item[1]['TRAIN_LOG_LIKELIHOOD'],
        )[0]

    return best_models, pd.DataFrame(candidate_rows)


def _parameter_count(n_states, n_features):
    start_parameters = n_states - 1
    transition_parameters = n_states * (n_states - 1)
    mean_parameters = n_states * n_features
    covariance_parameters = n_states * n_features * (n_features + 1) // 2
    return start_parameters + transition_parameters + mean_parameters + covariance_parameters


def _minimum_bhattacharyya_distance(model):
    distances = []
    for left in range(model.n_components):
        for right in range(left + 1, model.n_components):
            covariance = (model.covars_[left] + model.covars_[right]) / 2
            difference = model.means_[left] - model.means_[right]
            quadratic = difference @ np.linalg.pinv(covariance) @ difference / 8
            sign_mid, logdet_mid = np.linalg.slogdet(covariance)
            sign_left, logdet_left = np.linalg.slogdet(model.covars_[left])
            sign_right, logdet_right = np.linalg.slogdet(model.covars_[right])
            if min(sign_mid, sign_left, sign_right) <= 0:
                continue
            determinant = 0.5 * (logdet_mid - 0.5 * (logdet_left + logdet_right))
            distances.append(float(quadratic + determinant))
    return min(distances) if distances else np.nan


def _emission_log_probabilities(model, observation):
    return np.array([
        multivariate_normal.logpdf(
            observation,
            mean=model.means_[state],
            cov=model.covars_[state],
            allow_singular=True,
        )
        for state in range(model.n_components)
    ])


def filtered_state_probability(model, observations):
    observations = np.asarray(observations, dtype=float)
    log_probability = np.log(np.clip(model.startprob_, 1e-300, None))
    for row, observation in enumerate(observations):
        if row > 0:
            log_probability = logsumexp(
                log_probability[:, None]
                + np.log(np.clip(model.transmat_, 1e-300, None)),
                axis=0,
            )
        log_probability = log_probability + _emission_log_probabilities(model, observation)
        log_probability = log_probability - logsumexp(log_probability)
    return np.exp(log_probability)


def one_step_probability_integral_transforms(model, observations, initial_state_probability):
    """Compute marginal one-step-ahead PIT values and return the final filter."""
    observations = np.asarray(observations, dtype=float)
    filtered = np.asarray(initial_state_probability, dtype=float)
    pits = np.zeros_like(observations, dtype=float)

    for row, observation in enumerate(observations):
        prior = filtered @ model.transmat_
        standard_deviation = np.sqrt(np.maximum(
            np.diagonal(model.covars_, axis1=1, axis2=2),
            1e-12,
        ))
        marginal_cdf = norm.cdf(
            (observation[None, :] - model.means_) / standard_deviation
        )
        pits[row] = prior @ marginal_cdf

        log_filtered = np.log(np.clip(prior, 1e-300, None))
        log_filtered += _emission_log_probabilities(model, observation)
        log_filtered -= logsumexp(log_filtered)
        filtered = np.exp(log_filtered)

    return np.clip(pits, 1e-8, 1 - 1e-8), filtered


def residual_diagnostic_rows(n_states, split_name, pits, feature_names, alpha=SIGNIFICANCE_LEVEL):
    rows = []
    for column, feature in enumerate(feature_names):
        values = pits[:, column]
        ks_statistic, ks_p_value = kstest(values, 'uniform')
        normal_scores = norm.ppf(values)
        lag_10 = max(1, min(10, len(normal_scores) // 5))
        lag_20 = max(1, min(20, len(normal_scores) // 5))
        lags = sorted({lag_10, lag_20})
        ljung_box = acorr_ljungbox(normal_scores, lags=lags, return_df=True)
        squared_scores = normal_scores ** 2 - np.mean(normal_scores ** 2)
        squared_ljung_box = acorr_ljungbox(squared_scores, lags=lags, return_df=True)
        arch_statistic, arch_p_value, _, _ = het_arch(normal_scores, nlags=lag_10)

        ljung_box_statistic = float(ljung_box.loc[lag_10, 'lb_stat'])
        ljung_box_p_value = float(ljung_box.loc[lag_10, 'lb_pvalue'])
        ljung_box_20_statistic = float(ljung_box.loc[lag_20, 'lb_stat'])
        ljung_box_20_p_value = float(ljung_box.loc[lag_20, 'lb_pvalue'])
        squared_10_statistic = float(squared_ljung_box.loc[lag_10, 'lb_stat'])
        squared_10_p_value = float(squared_ljung_box.loc[lag_10, 'lb_pvalue'])
        squared_20_statistic = float(squared_ljung_box.loc[lag_20, 'lb_stat'])
        squared_20_p_value = float(squared_ljung_box.loc[lag_20, 'lb_pvalue'])
        volatility_pass = bool(
            squared_10_p_value >= alpha
            and squared_20_p_value >= alpha
            and arch_p_value >= alpha
        )
        rows.append({
            'N_STATES': n_states,
            'SPLIT': split_name,
            'FEATURE': feature,
            'N_OBSERVATIONS': len(values),
            'PIT_KS_STATISTIC': float(ks_statistic),
            'PIT_KS_P_VALUE': float(ks_p_value),
            'PIT_UNIFORM_AT_5_PERCENT': bool(ks_p_value >= alpha),
            'LJUNG_BOX_LAG': lag_10,
            'LJUNG_BOX_STATISTIC': ljung_box_statistic,
            'LJUNG_BOX_P_VALUE': ljung_box_p_value,
            'PIT_INDEPENDENT_AT_5_PERCENT': bool(ljung_box_p_value >= alpha),
            'LJUNG_BOX_LAG_20': lag_20,
            'LJUNG_BOX_LAG_20_STATISTIC': ljung_box_20_statistic,
            'LJUNG_BOX_LAG_20_P_VALUE': ljung_box_20_p_value,
            'PIT_INDEPENDENT_THROUGH_LAG_20_AT_5_PERCENT': bool(
                ljung_box_20_p_value >= alpha
            ),
            'SQUARED_PIT_LJUNG_BOX_LAG_10_STATISTIC': squared_10_statistic,
            'SQUARED_PIT_LJUNG_BOX_LAG_10_P_VALUE': squared_10_p_value,
            'SQUARED_PIT_LJUNG_BOX_LAG_20_STATISTIC': squared_20_statistic,
            'SQUARED_PIT_LJUNG_BOX_LAG_20_P_VALUE': squared_20_p_value,
            'ARCH_LM_LAG': lag_10,
            'ARCH_LM_STATISTIC': float(arch_statistic),
            'ARCH_LM_P_VALUE': float(arch_p_value),
            'NO_VOLATILITY_CLUSTERING_AT_5_PERCENT': volatility_pass,
            'PASS_BOTH_AT_5_PERCENT': bool(
                ks_p_value >= alpha and ljung_box_p_value >= alpha
            ),
            'PASS_ALL_RESIDUAL_TESTS_AT_5_PERCENT': bool(
                ks_p_value >= alpha
                and ljung_box_20_p_value >= alpha
                and volatility_pass
            ),
            'KS_NULL_HYPOTHESIS': 'one-step PIT is Uniform(0,1)',
            'LJUNG_BOX_NULL_HYPOTHESIS': (
                f'PIT normal scores have no autocorrelation through lag {lag_20}'
            ),
            'VOLATILITY_NULL_HYPOTHESIS': (
                'squared PIT normal scores have no autocorrelation and no ARCH effects'
            ),
        })
    return rows


def covariance_diagnostic_rows(
    n_states,
    model,
    feature_names,
    condition_threshold=DEFAULT_COVARIANCE_CONDITION_THRESHOLD,
):
    """Measure positive-definiteness and conditioning of each emission covariance."""
    spxt_column = list(feature_names).index('SPXT') if 'SPXT' in feature_names else 0
    ordered_states = np.argsort(model.means_[:, spxt_column])
    rows = []
    for regime, original_state in enumerate(ordered_states):
        covariance = np.asarray(model.covars_[original_state], dtype=float)
        symmetric = (covariance + covariance.T) / 2
        eigenvalues = np.linalg.eigvalsh(symmetric)
        condition_number = float(np.linalg.cond(symmetric))
        positive_definite = bool(np.all(eigenvalues > 0))
        finite_condition = bool(np.isfinite(condition_number))
        stable = bool(
            positive_definite
            and finite_condition
            and condition_number <= condition_threshold
        )
        rows.append({
            'N_STATES': n_states,
            'REGIME': regime,
            'ORIGINAL_STATE': int(original_state),
            'REGIME_ORDERING': 'ascending training-sample SPXT mean',
            'MIN_EIGENVALUE': float(eigenvalues.min()),
            'MAX_EIGENVALUE': float(eigenvalues.max()),
            'COVARIANCE_CONDITION_NUMBER': condition_number,
            'CONDITION_NUMBER_THRESHOLD': float(condition_threshold),
            'POSITIVE_DEFINITE': positive_definite,
            'FINITE_CONDITION_NUMBER': finite_condition,
            'NUMERICALLY_STABLE': stable,
            'UNSTABLE_REASON': (
                'none'
                if stable
                else (
                    'covariance_not_positive_definite'
                    if not positive_definite
                    else 'covariance_condition_number_exceeds_threshold'
                )
            ),
        })
    return rows


def emission_normality_rows(
    n_states,
    model,
    train_data,
    feature_names,
    alpha=SIGNIFICANCE_LEVEL,
):
    """Jarque-Bera checks of the Gaussian emission assumption within hard regimes."""
    values = np.asarray(train_data, dtype=float)
    states = model.predict(values)
    spxt_column = list(feature_names).index('SPXT') if 'SPXT' in feature_names else 0
    ordered_states = np.argsort(model.means_[:, spxt_column])
    corrected_alpha = alpha / len(feature_names)
    rows = []
    for regime, original_state in enumerate(ordered_states):
        regime_values = values[states == original_state]
        for column, feature in enumerate(feature_names):
            if len(regime_values) < 8:
                statistic = np.nan
                p_value = np.nan
            else:
                result = jarque_bera(regime_values[:, column])
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            reliable = bool(np.isfinite(p_value))
            rows.append({
                'N_STATES': n_states,
                'REGIME': regime,
                'ORIGINAL_STATE': int(original_state),
                'REGIME_ORDERING': 'ascending training-sample SPXT mean',
                'FEATURE': feature,
                'N_OBSERVATIONS': len(regime_values),
                'JARQUE_BERA_STATISTIC': statistic,
                'JARQUE_BERA_P_VALUE': p_value,
                'RELIABLE_FOR_5_PERCENT_DECISION': reliable,
                'NORMAL_AT_5_PERCENT': bool(reliable and p_value >= alpha),
                'BONFERRONI_ALPHA_WITHIN_REGIME': corrected_alpha,
                'NORMAL_AFTER_BONFERRONI_AT_5_PERCENT': bool(
                    reliable and p_value >= corrected_alpha
                ),
                'NULL_HYPOTHESIS': 'regime-conditional feature values are normally distributed',
            })
    return rows


def _complete_spell_lengths(states, state):
    states = np.asarray(states, dtype=int)
    if len(states) == 0:
        return np.array([], dtype=int)
    boundaries = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
    lengths = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start > 0 and stop < len(states) and states[start] == state:
            lengths.append(stop - start)
    return np.asarray(lengths, dtype=int)


def _geometric_cdf_discrepancy(lengths, exit_probability):
    support = np.unique(lengths)
    empirical = np.array([(lengths <= value).mean() for value in support])
    theoretical = 1 - (1 - exit_probability) ** support
    return float(np.max(np.abs(empirical - theoretical)))


def duration_geometric_rows(
    n_states,
    model,
    train_data,
    feature_names,
    n_bootstrap=DEFAULT_DURATION_BOOTSTRAP_REPLICATES,
    random_seed=DEFAULT_RANDOM_SEED,
    alpha=SIGNIFICANCE_LEVEL,
):
    """Bootstrap comparison of complete state spells with HMM-implied geometric durations."""
    values = np.asarray(train_data, dtype=float)
    states = model.predict(values)
    spxt_column = list(feature_names).index('SPXT') if 'SPXT' in feature_names else 0
    ordered_states = np.argsort(model.means_[:, spxt_column])
    rng = np.random.default_rng(random_seed + 1000 * n_states)
    rows = []

    for regime, original_state in enumerate(ordered_states):
        lengths = _complete_spell_lengths(states, original_state)
        self_transition = float(model.transmat_[original_state, original_state])
        exit_probability = float(np.clip(1 - self_transition, 1e-8, 1.0))
        reliable = len(lengths) >= 10
        if reliable:
            statistic = _geometric_cdf_discrepancy(lengths, exit_probability)
            bootstrap_statistics = [
                _geometric_cdf_discrepancy(
                    rng.geometric(exit_probability, size=len(lengths)),
                    exit_probability,
                )
                for _ in range(int(n_bootstrap))
            ]
            exceedances = sum(value >= statistic for value in bootstrap_statistics)
            p_value = (1 + exceedances) / (1 + len(bootstrap_statistics))
        else:
            statistic = np.nan
            p_value = np.nan
        rows.append({
            'N_STATES': n_states,
            'REGIME': regime,
            'ORIGINAL_STATE': int(original_state),
            'REGIME_ORDERING': 'ascending training-sample SPXT mean',
            'COMPLETE_SPELL_COUNT': len(lengths),
            'EMPIRICAL_MEAN_DURATION_DAYS': (
                float(lengths.mean()) if len(lengths) else np.nan
            ),
            'EMPIRICAL_MEDIAN_DURATION_DAYS': (
                float(np.median(lengths)) if len(lengths) else np.nan
            ),
            'EMPIRICAL_MAX_DURATION_DAYS': (
                int(lengths.max()) if len(lengths) else np.nan
            ),
            'SELF_TRANSITION_PROBABILITY': self_transition,
            'HMM_EXPECTED_DURATION_DAYS': 1 / exit_probability,
            'GEOMETRIC_CDF_DISCREPANCY': statistic,
            'BOOTSTRAP_REPLICATES': int(n_bootstrap),
            'BOOTSTRAP_P_VALUE': p_value,
            'RELIABLE_FOR_5_PERCENT_DECISION': reliable,
            'GEOMETRIC_DURATION_AT_5_PERCENT': bool(
                reliable and np.isfinite(p_value) and p_value >= alpha
            ),
            'NULL_HYPOTHESIS': (
                'complete inferred state-spell durations follow the HMM-implied geometric distribution'
            ),
        })
    return rows


def _transition_count_cube(states, n_states, n_segments=3):
    states = np.asarray(states, dtype=int)
    transition_indices = np.array_split(np.arange(len(states) - 1), n_segments)
    counts = np.zeros((n_segments, n_states, n_states), dtype=int)
    for segment, indices in enumerate(transition_indices):
        for index in indices:
            counts[segment, states[index], states[index + 1]] += 1
    return counts


def transition_homogeneity_test(counts):
    """Likelihood-ratio homogeneity test after removing empty rows/columns."""
    table = np.asarray(counts, dtype=float)
    table = table[table.sum(axis=1) > 0]
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan, 0
    statistic, p_value, degrees_freedom, _ = chi2_contingency(
        table,
        correction=False,
        lambda_='log-likelihood',
    )
    return float(statistic), float(p_value), int(degrees_freedom)


def _emission_mean_stability_test(data, states, state):
    data = np.asarray(data, dtype=float)
    states = np.asarray(states, dtype=int)
    midpoint = len(data) // 2
    first = data[:midpoint][states[:midpoint] == state]
    second = data[midpoint:][states[midpoint:] == state]
    if min(len(first), len(second)) <= data.shape[1] + 2:
        return np.nan, np.nan, 0, len(first), len(second)

    difference = first.mean(axis=0) - second.mean(axis=0)
    mean_covariance = np.cov(first, rowvar=False) / len(first)
    mean_covariance += np.cov(second, rowvar=False) / len(second)
    rank = int(np.linalg.matrix_rank(mean_covariance))
    statistic = float(difference @ np.linalg.pinv(mean_covariance) @ difference)
    p_value = float(chi2.sf(statistic, rank)) if rank > 0 else np.nan
    return statistic, p_value, rank, len(first), len(second)


def regime_stability_rows(
    n_states,
    model,
    train_data,
    feature_names,
    alpha=SIGNIFICANCE_LEVEL,
):
    values = np.asarray(train_data, dtype=float)
    states = model.predict(values)
    count_cube = _transition_count_cube(states, n_states)
    global_counts = count_cube.reshape(count_cube.shape[0], -1)
    global_statistic, global_p_value, global_df = transition_homogeneity_test(global_counts)
    spxt_column = list(feature_names).index('SPXT') if 'SPXT' in feature_names else 0
    ordered_states = np.argsort(model.means_[:, spxt_column])
    rows = []

    for regime, original_state in enumerate(ordered_states):
        count = int((states == original_state).sum())
        occupancy = count / len(states)
        self_transition = float(model.transmat_[original_state, original_state])
        expected_duration = 1.0 / max(1.0 - self_transition, 1e-12)
        transition_statistic, transition_p_value, transition_df = transition_homogeneity_test(
            count_cube[:, original_state, :]
        )
        emission_statistic, emission_p_value, emission_df, first_count, second_count = (
            _emission_mean_stability_test(values, states, original_state)
        )

        segment_occupancies = []
        for segment_indices in np.array_split(np.arange(len(states)), 3):
            segment_occupancies.append(float((states[segment_indices] == original_state).mean()))

        reasons = []
        if occupancy < 0.05:
            reasons.append('rare_occupancy_below_5_percent')
        if expected_duration < 2:
            reasons.append('expected_duration_below_2_days')
        if np.isfinite(transition_p_value) and transition_p_value < alpha:
            reasons.append('transition_distribution_changes_over_time')
        if not np.isfinite(transition_p_value):
            reasons.append('insufficient_transitions_for_stability_test')
        if np.isfinite(emission_p_value) and emission_p_value < alpha:
            reasons.append('emission_mean_changes_between_train_halves')
        if not np.isfinite(emission_p_value):
            reasons.append('insufficient_observations_for_emission_stability_test')

        rows.append({
            'N_STATES': n_states,
            'REGIME': regime,
            'ORIGINAL_STATE': int(original_state),
            'REGIME_ORDERING': 'ascending training-sample SPXT mean',
            'TRAIN_COUNT': count,
            'TRAIN_OCCUPANCY': occupancy,
            'SEGMENT_1_OCCUPANCY': segment_occupancies[0],
            'SEGMENT_2_OCCUPANCY': segment_occupancies[1],
            'SEGMENT_3_OCCUPANCY': segment_occupancies[2],
            'SELF_TRANSITION_PROBABILITY': self_transition,
            'EXPECTED_DURATION_DAYS': expected_duration,
            'TRANSITION_STABILITY_G_STATISTIC': transition_statistic,
            'TRANSITION_STABILITY_DF': transition_df,
            'TRANSITION_STABILITY_P_VALUE': transition_p_value,
            'TRANSITION_STABLE_AT_5_PERCENT': bool(
                np.isfinite(transition_p_value) and transition_p_value >= alpha
            ),
            'EMISSION_MEAN_STABILITY_WALD_STATISTIC': emission_statistic,
            'EMISSION_MEAN_STABILITY_DF': emission_df,
            'EMISSION_MEAN_STABILITY_P_VALUE': emission_p_value,
            'EMISSION_MEAN_STABLE_AT_5_PERCENT': bool(
                np.isfinite(emission_p_value) and emission_p_value >= alpha
            ),
            'FIRST_HALF_COUNT': first_count,
            'SECOND_HALF_COUNT': second_count,
            'GLOBAL_TRANSITION_STABILITY_G_STATISTIC': global_statistic,
            'GLOBAL_TRANSITION_STABILITY_DF': global_df,
            'GLOBAL_TRANSITION_STABILITY_P_VALUE': global_p_value,
            'UNSTABLE_FLAG': bool(reasons),
            'UNSTABLE_REASONS': '|'.join(reasons) if reasons else 'none',
            'TRANSITION_NULL_HYPOTHESIS': 'outgoing transition probabilities are constant across three train segments',
            'EMISSION_NULL_HYPOTHESIS': 'regime emission mean vector is equal across the two train halves',
        })
    return rows


def goodness_of_fit_row(n_states, model, train, validation, prediction):
    train_values = np.asarray(train, dtype=float)
    posterior = model.predict_proba(train_values)
    entropy = float(-(posterior * np.log(np.clip(posterior, 1e-300, None))).sum())
    hard_states = model.predict(train_values)
    counts = np.bincount(hard_states, minlength=n_states)
    train_ll = float(model.score(train_values))
    validation_ll = float(model.score(validation))
    prediction_ll = float(model.score(prediction))
    return {
        'N_STATES': n_states,
        'N_FEATURES': train_values.shape[1],
        'FEATURE_SET': 'common six exogenous return series',
        'DIAGNOSTIC_ONLY': True,
        'PARAMETER_COUNT': _parameter_count(n_states, train_values.shape[1]),
        'TRAIN_ROWS': len(train_values),
        'VALIDATION_ROWS': len(validation),
        'PREDICTION_ROWS': len(prediction),
        'TRAIN_LOG_LIKELIHOOD': train_ll,
        'TRAIN_LOG_LIKELIHOOD_PER_ROW': train_ll / len(train_values),
        'AIC': float(model.aic(train_values)),
        'BIC': float(model.bic(train_values)),
        'ICL_APPROX': float(model.bic(train_values) + 2 * entropy),
        'POSTERIOR_ENTROPY': entropy,
        'MEAN_POSTERIOR_ENTROPY': entropy / len(train_values),
        'VALIDATION_LOG_LIKELIHOOD': validation_ll,
        'VALIDATION_LOG_LIKELIHOOD_PER_ROW': validation_ll / len(validation),
        'PREDICTION_LOG_LIKELIHOOD': prediction_ll,
        'PREDICTION_LOG_LIKELIHOOD_PER_ROW': prediction_ll / len(prediction),
        'MIN_REGIME_COUNT': int(counts.min()),
        'MIN_REGIME_OCCUPANCY': float(counts.min() / len(train_values)),
        'MIN_BHATTACHARYYA_DISTANCE': _minimum_bhattacharyya_distance(model),
        'CONVERGED': bool(model.monitor_.converged),
    }


def bootstrap_regime_count_tests(
    models,
    train_data,
    comparisons=((2, 3), (3, 4)),
    n_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    n_starts=DEFAULT_BOOTSTRAP_STARTS,
    random_seed=DEFAULT_RANDOM_SEED,
    alpha=SIGNIFICANCE_LEVEL,
    max_iter=1000,
):
    """Parametric-bootstrap LRT; ordinary chi-square asymptotics are invalid here."""
    values = np.asarray(train_data, dtype=float)
    rng = np.random.default_rng(random_seed)
    rows = []

    for lower, higher in comparisons:
        observed_statistic = max(
            0.0,
            2 * (models[higher].score(values) - models[lower].score(values)),
        )
        bootstrap_statistics = []
        failed_replicates = 0

        for _ in range(int(n_replicates)):
            sample_seed = int(rng.integers(0, 2 ** 31 - 1))
            sample, _ = models[lower].sample(len(values), random_state=sample_seed)
            fit_seed = int(rng.integers(0, 2 ** 31 - 1))
            try:
                fitted, _ = fit_multistart_hmms(
                    sample,
                    state_counts=(lower, higher),
                    n_starts=n_starts,
                    random_seed=fit_seed,
                    min_regime_count=1,
                    max_iter=max_iter,
                )
                statistic = max(
                    0.0,
                    2 * (fitted[higher].score(sample) - fitted[lower].score(sample)),
                )
                bootstrap_statistics.append(statistic)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                failed_replicates += 1

        if bootstrap_statistics:
            exceedances = sum(value >= observed_statistic for value in bootstrap_statistics)
            p_value = (1 + exceedances) / (1 + len(bootstrap_statistics))
            critical_value = float(np.quantile(bootstrap_statistics, 1 - alpha))
        else:
            p_value = np.nan
            critical_value = np.nan

        reject = bool(np.isfinite(p_value) and p_value < alpha)
        minimum_valid = max(19, int(np.ceil(0.8 * int(n_replicates))))
        reliable = len(bootstrap_statistics) >= minimum_valid
        rows.append({
            'LOWER_N_STATES': lower,
            'HIGHER_N_STATES': higher,
            'OBSERVED_LR_STATISTIC': observed_statistic,
            'BOOTSTRAP_REPLICATES_REQUESTED': int(n_replicates),
            'BOOTSTRAP_REPLICATES_VALID': len(bootstrap_statistics),
            'BOOTSTRAP_REPLICATES_FAILED': failed_replicates,
            'BOOTSTRAP_95_PERCENT_CRITICAL_VALUE': critical_value,
            'BOOTSTRAP_P_VALUE': p_value,
            'MINIMUM_VALID_REPLICATES_FOR_DECISION': minimum_valid,
            'RELIABLE_FOR_5_PERCENT_DECISION': reliable,
            'REJECT_LOWER_MODEL_AT_5_PERCENT': reject,
            'SUPPORTED_MODEL': (higher if reject else lower) if reliable else np.nan,
            'NULL_HYPOTHESIS': f'q{lower} generates the data; q{higher} does not add a required regime',
            'INTERPRETATION': (
                'too few successful bootstrap refits for a reliable decision'
                if not reliable
                else (f'evidence favors q{higher}' if reject
                      else f'insufficient evidence to move beyond q{lower}')
            ),
        })

    return pd.DataFrame(rows)


def build_recommendation_table(
    gof,
    stability,
    bootstrap,
    covariance=None,
    durations=None,
):
    bic_choice = int(gof.loc[gof['BIC'].idxmin(), 'N_STATES'])
    icl_choice = int(gof.loc[gof['ICL_APPROX'].idxmin(), 'N_STATES'])
    validation_choice = int(
        gof.loc[gof['VALIDATION_LOG_LIKELIHOOD_PER_ROW'].idxmax(), 'N_STATES']
    )
    bootstrap_choice = 2
    bootstrap_reliable = True
    for lower, higher in [(2, 3), (3, 4)]:
        comparison = bootstrap[
            (bootstrap['LOWER_N_STATES'] == lower)
            & (bootstrap['HIGHER_N_STATES'] == higher)
        ]
        if bootstrap_choice == lower and not comparison.empty:
            if not bool(comparison.iloc[0]['RELIABLE_FOR_5_PERCENT_DECISION']):
                bootstrap_reliable = False
                break
            if bool(comparison.iloc[0]['REJECT_LOWER_MODEL_AT_5_PERCENT']):
                bootstrap_choice = higher

    rows = []
    for _, fit in gof.iterrows():
        n_states = int(fit['N_STATES'])
        unstable_count = int(
            stability[stability['N_STATES'] == n_states]['UNSTABLE_FLAG'].sum()
        )
        numerical_count = 0 if covariance is None else int(
            (~covariance[
                covariance['N_STATES'] == n_states
            ]['NUMERICALLY_STABLE']).sum()
        )
        non_geometric_count = 0 if durations is None else int(
            (
                durations[durations['N_STATES'] == n_states][
                    'RELIABLE_FOR_5_PERCENT_DECISION'
                ]
                & ~durations[durations['N_STATES'] == n_states][
                    'GEOMETRIC_DURATION_AT_5_PERCENT'
                ]
            ).sum()
        )
        support = {
            'BIC': n_states == bic_choice,
            'ICL': n_states == icl_choice,
            'VALIDATION_LIKELIHOOD': n_states == validation_choice,
            'BOOTSTRAP_LRT': bootstrap_reliable and n_states == bootstrap_choice,
        }
        rows.append({
            'N_STATES': n_states,
            'BIC': fit['BIC'],
            'ICL_APPROX': fit['ICL_APPROX'],
            'VALIDATION_LOG_LIKELIHOOD_PER_ROW': fit['VALIDATION_LOG_LIKELIHOOD_PER_ROW'],
            'UNSTABLE_REGIME_COUNT': unstable_count,
            'NUMERICALLY_UNSTABLE_REGIME_COUNT': numerical_count,
            'NON_GEOMETRIC_DURATION_REGIME_COUNT': non_geometric_count,
            'BIC_SUPPORTS': support['BIC'],
            'ICL_SUPPORTS': support['ICL'],
            'VALIDATION_LIKELIHOOD_SUPPORTS': support['VALIDATION_LIKELIHOOD'],
            'BOOTSTRAP_LRT_SUPPORTS': support['BOOTSTRAP_LRT'],
            'SUPPORT_CRITERIA_COUNT': sum(support.values()),
            'SUPPORTING_CRITERIA': '|'.join(name for name, value in support.items() if value),
        })

    recommendation = pd.DataFrame(rows)
    winner = recommendation.sort_values(
        [
            'SUPPORT_CRITERIA_COUNT',
            'NUMERICALLY_UNSTABLE_REGIME_COUNT',
            'UNSTABLE_REGIME_COUNT',
            'NON_GEOMETRIC_DURATION_REGIME_COUNT',
            'BIC',
            'N_STATES',
        ],
        ascending=[False, True, True, True, True, True],
    ).iloc[0]['N_STATES']
    recommendation['RECOMMENDED'] = recommendation['N_STATES'] == int(winner)
    recommendation['DECISION_RULE'] = (
        'most supporting criteria; then fewer numerically unstable, generally unstable, '
        'and non-geometric regimes; then lower BIC; then fewer states'
    )
    return recommendation


def plot_model_selection(gof, stability, output_path):
    state_counts = gof['N_STATES'].astype(int).to_numpy()
    unstable = [
        int(stability[stability['N_STATES'] == n_states]['UNSTABLE_FLAG'].sum())
        for n_states in state_counts
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.flatten()
    axes[0].plot(state_counts, gof['BIC'], marker='o', label='BIC')
    axes[0].plot(state_counts, gof['ICL_APPROX'], marker='o', label='ICL approx.')
    axes[0].set_title('In-sample penalized fit; lower is better')
    axes[0].legend()
    axes[1].plot(
        state_counts,
        gof['VALIDATION_LOG_LIKELIHOOD_PER_ROW'],
        marker='o',
        color='#167D8D',
    )
    axes[1].set_title('Validation log-likelihood per row; higher is better')
    axes[2].plot(
        state_counts,
        gof['MIN_BHATTACHARYYA_DISTANCE'],
        marker='o',
        color='#D17A22',
    )
    axes[2].set_title('Minimum regime separation; higher is better')
    axes[3].bar(state_counts, unstable, color='#9B3A3A')
    axes[3].set_title('Regimes flagged as unstable; lower is better')
    for ax in axes:
        ax.set_xlabel('Number of regimes')
        ax.set_xticks(state_counts)
        ax.grid(alpha=0.2)
    fig.suptitle('HMM q2/q3/q4 diagnostics on the same six-series sample')
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_assumption_diagnostics(covariance, normality, residuals, durations, output_path):
    """Visual summary of the additive numerical and distributional checks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    covariance_plot = covariance.copy()
    finite = covariance_plot.loc[
        np.isfinite(covariance_plot['COVARIANCE_CONDITION_NUMBER']),
        'COVARIANCE_CONDITION_NUMBER',
    ]
    infinity_height = max(float(finite.max()) if len(finite) else 1.0, 1.0) * 10
    covariance_plot['PLOT_CONDITION_NUMBER'] = covariance_plot[
        'COVARIANCE_CONDITION_NUMBER'
    ].replace([np.inf, -np.inf], infinity_height)
    labels = [
        f"q{int(row.N_STATES)} r{int(row.REGIME)}"
        for row in covariance_plot.itertuples()
    ]
    colors = [
        '#167D8D' if stable else '#9B3A3A'
        for stable in covariance_plot['NUMERICALLY_STABLE']
    ]
    axes[0].bar(labels, covariance_plot['PLOT_CONDITION_NUMBER'], color=colors)
    threshold = float(covariance_plot['CONDITION_NUMBER_THRESHOLD'].iloc[0])
    axes[0].axhline(threshold, color='#9B3A3A', linestyle='--', label='stability threshold')
    axes[0].set_yscale('log')
    axes[0].set_title('Emission covariance condition numbers')
    axes[0].set_ylabel('Condition number (log scale)')
    axes[0].legend()

    reliable_normality = normality[
        normality['RELIABLE_FOR_5_PERCENT_DECISION']
    ]
    normality_summary = reliable_normality.groupby('N_STATES').agg(
        TESTS=('JARQUE_BERA_P_VALUE', 'count'),
        REJECTIONS=('NORMAL_AFTER_BONFERRONI_AT_5_PERCENT', lambda values: (~values).sum()),
    ).reset_index()
    normality_summary['REJECTION_RATE'] = (
        normality_summary['REJECTIONS'] / normality_summary['TESTS']
    )
    axes[1].bar(
        normality_summary['N_STATES'].astype(str),
        normality_summary['REJECTION_RATE'],
        color='#D17A22',
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Gaussian emission tests rejected')
    axes[1].set_xlabel('Number of regimes')
    axes[1].set_ylabel('Bonferroni-adjusted rejection rate')

    validation = residuals[residuals['SPLIT'] == 'validation'].copy()
    volatility_summary = validation.groupby('N_STATES')[
        'NO_VOLATILITY_CLUSTERING_AT_5_PERCENT'
    ].apply(lambda values: (~values).mean())
    axes[2].bar(
        volatility_summary.index.astype(str),
        volatility_summary.to_numpy(),
        color='#6D597A',
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Validation volatility diagnostics failed')
    axes[2].set_xlabel('Number of regimes')
    axes[2].set_ylabel('Feature failure rate')

    reliable = durations[durations['RELIABLE_FOR_5_PERCENT_DECISION']].copy()
    duration_labels = [
        f"q{int(row.N_STATES)} r{int(row.REGIME)}"
        for row in reliable.itertuples()
    ]
    axes[3].scatter(
        duration_labels,
        reliable['BOOTSTRAP_P_VALUE'],
        color='#167D8D',
        s=45,
    )
    axes[3].axhline(
        SIGNIFICANCE_LEVEL,
        color='#9B3A3A',
        linestyle='--',
        label='5% significance level',
    )
    axes[3].set_ylim(0, 1)
    axes[3].set_title('Geometric-duration bootstrap tests')
    axes[3].set_ylabel('Bootstrap p-value')
    axes[3].legend()

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('HMM numerical, emission, volatility, and duration diagnostics')
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def run_hmm_diagnostics(
    n_starts=DEFAULT_N_STARTS,
    bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_starts=DEFAULT_BOOTSTRAP_STARTS,
    random_seed=DEFAULT_RANDOM_SEED,
    output_dir=None,
):
    output_dir = Path(output_dir) if output_dir is not None else BASE_DIR / 'phase_1'
    output_dir.mkdir(parents=True, exist_ok=True)
    exog_rets = pd.read_csv(
        BASE_DIR / 'processed' / 'exog_rets.csv',
        index_col=[0],
        parse_dates=[0],
    )
    missing_features = set(DIAGNOSTIC_FEATURES) - set(exog_rets.columns)
    if missing_features:
        raise ValueError(
            f'HMM diagnostics are missing required features: {sorted(missing_features)}'
        )
    exog_rets = exog_rets[list(DIAGNOSTIC_FEATURES)].dropna()
    train = exog_rets[exog_rets.index < TRAIN_END]
    validation = exog_rets[
        (exog_rets.index >= TRAIN_END) & (exog_rets.index < VALIDATION_END)
    ]
    prediction = exog_rets[exog_rets.index >= VALIDATION_END]
    if min(len(train), len(validation), len(prediction)) == 0:
        raise ValueError('Train, validation, and prediction samples must all be non-empty.')

    train_mean = train.mean()
    train_scale = train.std(ddof=0).replace(0, 1.0)
    train_scaled = (train - train_mean) / train_scale
    validation_scaled = (validation - train_mean) / train_scale
    prediction_scaled = (prediction - train_mean) / train_scale

    models, multistart = fit_multistart_hmms(
        train_scaled,
        n_starts=n_starts,
        random_seed=random_seed,
    )
    multistart_path = output_dir / 'hmm_diagnostic_multistart.csv'
    multistart.to_csv(multistart_path, index=False)

    gof_rows = []
    residual_rows = []
    stability_rows = []
    covariance_rows = []
    normality_rows = []
    duration_rows = []
    for n_states, model in models.items():
        gof_rows.append(goodness_of_fit_row(
            n_states,
            model,
            train_scaled,
            validation_scaled,
            prediction_scaled,
        ))
        initial = filtered_state_probability(model, train_scaled)
        validation_pits, validation_filter = one_step_probability_integral_transforms(
            model,
            validation_scaled,
            initial,
        )
        prediction_pits, _ = one_step_probability_integral_transforms(
            model,
            prediction_scaled,
            validation_filter,
        )
        residual_rows.extend(residual_diagnostic_rows(
            n_states,
            'validation',
            validation_pits,
            exog_rets.columns,
        ))
        residual_rows.extend(residual_diagnostic_rows(
            n_states,
            'prediction',
            prediction_pits,
            exog_rets.columns,
        ))
        stability_rows.extend(regime_stability_rows(
            n_states,
            model,
            train_scaled,
            exog_rets.columns,
        ))
        covariance_rows.extend(covariance_diagnostic_rows(
            n_states,
            model,
            exog_rets.columns,
        ))
        normality_rows.extend(emission_normality_rows(
            n_states,
            model,
            train_scaled,
            exog_rets.columns,
        ))
        duration_rows.extend(duration_geometric_rows(
            n_states,
            model,
            train_scaled,
            exog_rets.columns,
            random_seed=random_seed,
        ))

    gof = pd.DataFrame(gof_rows).sort_values('N_STATES')
    residuals = pd.DataFrame(residual_rows)
    stability = pd.DataFrame(stability_rows)
    covariance = pd.DataFrame(covariance_rows)
    normality = pd.DataFrame(normality_rows)
    durations = pd.DataFrame(duration_rows)
    for n_states in DIAGNOSTIC_STATES:
        failed = residuals[
            (residuals['N_STATES'] == n_states)
            & (residuals['SPLIT'] == 'validation')
            & ~residuals['PASS_BOTH_AT_5_PERCENT']
        ]
        gof.loc[gof['N_STATES'] == n_states, 'VALIDATION_RESIDUAL_TESTS_FAILED'] = len(failed)
        gof.loc[gof['N_STATES'] == n_states, 'UNSTABLE_REGIME_COUNT'] = int(
            stability[stability['N_STATES'] == n_states]['UNSTABLE_FLAG'].sum()
        )
        volatility_failed = residuals[
            (residuals['N_STATES'] == n_states)
            & (residuals['SPLIT'] == 'validation')
            & ~residuals['NO_VOLATILITY_CLUSTERING_AT_5_PERCENT']
        ]
        gof.loc[
            gof['N_STATES'] == n_states,
            'VALIDATION_VOLATILITY_TESTS_FAILED',
        ] = len(volatility_failed)
        gof.loc[
            gof['N_STATES'] == n_states,
            'NUMERICALLY_UNSTABLE_REGIME_COUNT',
        ] = int((~covariance[
            covariance['N_STATES'] == n_states
        ]['NUMERICALLY_STABLE']).sum())
        reliable_normality = normality[
            (normality['N_STATES'] == n_states)
            & normality['RELIABLE_FOR_5_PERCENT_DECISION']
        ]
        gof.loc[
            gof['N_STATES'] == n_states,
            'EMISSION_NORMALITY_TESTS_REJECTED',
        ] = int((~reliable_normality[
            'NORMAL_AFTER_BONFERRONI_AT_5_PERCENT'
        ]).sum())
        duration_subset = durations[
            (durations['N_STATES'] == n_states)
            & durations['RELIABLE_FOR_5_PERCENT_DECISION']
        ]
        gof.loc[
            gof['N_STATES'] == n_states,
            'NON_GEOMETRIC_DURATION_REGIME_COUNT',
        ] = int((~duration_subset['GEOMETRIC_DURATION_AT_5_PERCENT']).sum())

    bootstrap = bootstrap_regime_count_tests(
        models,
        train_scaled,
        n_replicates=bootstrap_replicates,
        n_starts=bootstrap_starts,
        random_seed=random_seed,
    )
    recommendation = build_recommendation_table(
        gof,
        stability,
        bootstrap,
        covariance=covariance,
        durations=durations,
    )

    paths = {
        'gof': output_dir / 'hmm_gof_summary.csv',
        'residuals': output_dir / 'hmm_residual_diagnostics.csv',
        'stability': output_dir / 'hmm_regime_stability.csv',
        'covariance': output_dir / 'hmm_covariance_diagnostics.csv',
        'normality': output_dir / 'hmm_emission_normality.csv',
        'durations': output_dir / 'hmm_duration_geometric_diagnostics.csv',
        'bootstrap': output_dir / 'hmm_regime_count_bootstrap.csv',
        'recommendation': output_dir / 'hmm_regime_count_recommendation.csv',
        'plot': output_dir / 'hmm_diagnostics_model_selection.png',
        'assumption_plot': output_dir / 'hmm_diagnostics_assumption_checks.png',
        'multistart': multistart_path,
        'config': output_dir / 'hmm_diagnostic_run_config.csv',
    }
    gof.to_csv(paths['gof'], index=False)
    residuals.to_csv(paths['residuals'], index=False)
    stability.to_csv(paths['stability'], index=False)
    covariance.to_csv(paths['covariance'], index=False)
    normality.to_csv(paths['normality'], index=False)
    durations.to_csv(paths['durations'], index=False)
    bootstrap.to_csv(paths['bootstrap'], index=False)
    recommendation.to_csv(paths['recommendation'], index=False)
    pd.DataFrame([{
        'TRAIN_START': train.index.min().date(),
        'TRAIN_END_EXCLUSIVE': TRAIN_END,
        'VALIDATION_START': validation.index.min().date(),
        'VALIDATION_END_EXCLUSIVE': VALIDATION_END,
        'PREDICTION_START': prediction.index.min().date(),
        'PREDICTION_END': prediction.index.max().date(),
        'FEATURES': '|'.join(exog_rets.columns),
        'STANDARDIZATION': 'train mean and population standard deviation',
        'N_STARTS': int(n_starts),
        'BOOTSTRAP_REPLICATES': int(bootstrap_replicates),
        'BOOTSTRAP_STARTS': int(bootstrap_starts),
        'RANDOM_SEED': int(random_seed),
        'SIGNIFICANCE_LEVEL': SIGNIFICANCE_LEVEL,
        'COVARIANCE_CONDITION_THRESHOLD': DEFAULT_COVARIANCE_CONDITION_THRESHOLD,
        'DURATION_BOOTSTRAP_REPLICATES': DEFAULT_DURATION_BOOTSTRAP_REPLICATES,
        'DIAGNOSTIC_MODELS_USED_DOWNSTREAM': False,
    }]).to_csv(paths['config'], index=False)
    plot_model_selection(gof, stability, paths['plot'])
    plot_assumption_diagnostics(
        covariance,
        normality,
        residuals,
        durations,
        paths['assumption_plot'],
    )
    return {
        'models': models,
        'gof': gof,
        'residuals': residuals,
        'stability': stability,
        'covariance': covariance,
        'normality': normality,
        'durations': durations,
        'bootstrap': bootstrap,
        'recommendation': recommendation,
        'paths': paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starts', type=int, default=DEFAULT_N_STARTS)
    parser.add_argument(
        '--bootstrap-replicates',
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument('--bootstrap-starts', type=int, default=DEFAULT_BOOTSTRAP_STARTS)
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    result = run_hmm_diagnostics(
        n_starts=args.starts,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_starts=args.bootstrap_starts,
        random_seed=args.seed,
    )
    print(result['recommendation'].to_string(index=False))
    print('Diagnostic outputs:')
    for path in result['paths'].values():
        print(f'- {path}')


if __name__ == '__main__':
    main()
