import unittest

import numpy as np
import pandas as pd

from p1_9_transition_feature_ablation import (
    feature_group,
    multiclass_expected_calibration_error,
    probability_metrics,
    state_conditional_block_permutation,
    summarize_permutations,
)


class TestTransitionFeatureAblation(unittest.TestCase):
    def test_feature_groups_are_mutually_exclusive(self):
        self.assertEqual(feature_group('STATE_LAG_0_2'), 'state')
        self.assertEqual(feature_group('DURATION_STATE_2'), 'duration')
        self.assertEqual(feature_group('SPXT_LAG_0'), 'market')

    def test_state_conditional_permutation_does_not_cross_states(self):
        frame = pd.DataFrame({
            'MARKET_A': np.arange(12, dtype=float),
            'MARKET_B': np.arange(12, dtype=float) + 100,
            'UNCHANGED': np.arange(12, dtype=float) + 200,
        })
        states = np.repeat([0, 1], 6)
        result = state_conditional_block_permutation(
            frame,
            ['MARKET_A', 'MARKET_B'],
            states,
            block_length=2,
            rng=np.random.default_rng(7),
        )

        for state in [0, 1]:
            mask = states == state
            self.assertEqual(
                set(result.loc[mask, 'MARKET_A']),
                set(frame.loc[mask, 'MARKET_A']),
            )
        np.testing.assert_array_equal(result['UNCHANGED'], frame['UNCHANGED'])
        np.testing.assert_array_equal(result['MARKET_B'] - result['MARKET_A'], 100.0)

    def test_probability_metrics_and_ece_are_finite(self):
        y = np.array([0, 1, 1, 0])
        current = np.array([0, 0, 1, 1])
        probabilities = np.array([
            [0.9, 0.1],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.6, 0.4],
        ])
        metrics = probability_metrics(y, current, probabilities, [0, 1])
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreaterEqual(multiclass_expected_calibration_error(y, probabilities, [0, 1]), 0)

    def test_permutation_summary_reports_one_sided_p_value(self):
        replicates = pd.DataFrame({
            'N_STATES': [2] * 4,
            'MODEL': ['MLG'] * 4,
            'SPLIT': ['validation'] * 4,
            'FEATURE_GROUP': ['duration'] * 4,
            'BLOCK_LENGTH': [20] * 4,
            'GROUP_COLUMN_COUNT': [2] * 4,
            'DELTA_LOG_LOSS': [0.1, 0.2, -0.1, 0.3],
        })
        summary = summarize_permutations(replicates).iloc[0]
        self.assertAlmostEqual(summary['DELTA_LOG_LOSS_MEAN'], 0.125)
        self.assertAlmostEqual(summary['DELTA_LOG_LOSS_ONE_SIDED_P_VALUE'], 0.4)


if __name__ == '__main__':
    unittest.main()
