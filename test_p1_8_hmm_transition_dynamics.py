import os
import unittest

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from p1_8_hmm_transition_dynamics import (
    build_event_observations,
    compare_abrupt_and_gradual_forecasts,
    posterior_frame,
    rolling_moment_frame,
    transition_monotonicity_tests,
    windowed_filtered_probabilities,
)


class TestHMMTransitionDynamics(unittest.TestCase):
    @staticmethod
    def _known_hmm():
        model = GaussianHMM(n_components=2, covariance_type='full')
        model.startprob_ = np.array([0.7, 0.3])
        model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
        model.means_ = np.array([[-1.0], [1.0]])
        model.covars_ = np.array([[[0.4]], [[0.6]]])
        model.n_features = 1
        return model

    def test_windowed_filter_matches_last_forward_backward_probability(self):
        model = self._known_hmm()
        observations = np.array([[-1.2], [-0.7], [0.2], [1.1], [0.8], [-0.4]])
        window_size = 3
        probabilities = windowed_filtered_probabilities(model, observations, window_size)

        for end in range(len(observations)):
            start = max(0, end - window_size + 1)
            expected = model.predict_proba(observations[start:end + 1])[-1]
            np.testing.assert_allclose(probabilities[end], expected, atol=1e-10)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_rolling_moments_are_trailing_only(self):
        index = pd.date_range('2000-01-01', periods=8)
        observations = pd.DataFrame({'SPXT': [0, 1, 2, 3, 4, 5, 6, 100]}, index=index)
        before = rolling_moment_frame(observations, n_train=5, windows=(3,))
        changed = observations.copy()
        changed.iloc[-1, 0] = 1000
        after = rolling_moment_frame(changed, n_train=5, windows=(3,))

        pd.testing.assert_series_equal(before.iloc[:-1, 0], after.iloc[:-1, 0])
        pd.testing.assert_series_equal(before.iloc[:-1, 1], after.iloc[:-1, 1])
        self.assertNotEqual(before.iloc[-1, 0], after.iloc[-1, 0])

    def test_event_builder_and_monotonicity_detect_gradual_buildup(self):
        rows = []
        for event_id in range(1, 7):
            for relative_day, probability in zip(range(-3, 4), [0.1, 0.2, 0.3, 0.8, 0.85, 0.9, 0.92]):
                rows.append({
                    'N_STATES': 2,
                    'EVENT_ID': event_id,
                    'FROM_STATE': 0,
                    'TO_STATE': 1,
                    'RELATIVE_DAY': relative_day,
                    'DESTINATION_POSTERIOR': probability,
                    'MAX_POSTERIOR': max(probability, 1 - probability),
                })
        events = pd.DataFrame(rows)
        result = transition_monotonicity_tests(
            events,
            event_radius=3,
            min_events=5,
            n_bootstrap=49,
            random_seed=7,
        ).iloc[0]

        self.assertTrue(result['RELIABLE_FOR_5_PERCENT_DECISION'])
        self.assertTrue(result['GRADUAL_BUILDUP_SUPPORTED_AT_5_PERCENT'])
        self.assertGreater(result['BOOTSTRAP_SLOPE_CI_LOWER'], 0)
        self.assertEqual(result['SHARE_POSITIVE_PRE_TRANSITION_SLOPE'], 1.0)

    def test_event_observations_include_rolling_supporting_evidence(self):
        index = pd.date_range('2000-01-01', periods=30)
        states = pd.Series([0] * 10 + [1] * 10 + [0] * 10, index=index)
        probabilities = np.column_stack([
            np.where(states.to_numpy() == 0, 0.85, 0.15),
            np.where(states.to_numpy() == 1, 0.85, 0.15),
        ])
        posterior = posterior_frame(index, states, probabilities)
        moments = pd.DataFrame({
            'ROLLING_MEAN_Z_W5_SPXT': np.linspace(-1, 1, len(index)),
            'ROLLING_STD_RATIO_W5_SPXT': np.linspace(0.5, 1.5, len(index)),
        }, index=index)
        events, counts = build_event_observations(
            2,
            states,
            posterior,
            moments,
            event_radius=3,
        )

        self.assertEqual(len(counts), 2)
        self.assertIn('ROLLING_MEAN_Z_W5_SPXT', events.columns)
        self.assertEqual(sorted(events['RELATIVE_DAY'].unique()), list(range(-3, 4)))

    def test_abrupt_soft_forecast_comparison_exports_metrics_and_calibration(self):
        validation_dates = pd.date_range('2009-06-02', periods=40)
        prediction_dates = pd.date_range('2017-01-03', periods=40)
        index = validation_dates.append(prediction_dates)
        states = pd.Series(np.tile([0, 0, 0, 1, 1], 16), index=index)
        posterior = np.column_stack([
            np.where(states.to_numpy() == 0, 0.8, 0.2),
            np.where(states.to_numpy() == 1, 0.8, 0.2),
        ])
        transition = np.array([[0.8, 0.2], [0.3, 0.7]])

        report, paired, calibration = compare_abrupt_and_gradual_forecasts(
            2,
            states,
            posterior,
            transition,
            n_bootstrap=19,
            block_length=5,
            random_seed=11,
        )

        self.assertEqual(len(report), 8)
        self.assertEqual(len(paired), 8)
        self.assertFalse(calibration.empty)
        self.assertEqual(set(report['MODEL']), {
            'hard_state_fixed_transition',
            'soft_posterior_fixed_transition',
            'hard_state_jeffreys_smoothed_transition',
            'soft_posterior_jeffreys_smoothed_transition',
        })
        self.assertEqual(set(report['TRANSITION_MATRIX_VARIANT']), {'fixed', 'jeffreys_smoothed'})
        self.assertTrue(np.isfinite(report['LOG_LOSS']).all())


if __name__ == '__main__':
    unittest.main()
