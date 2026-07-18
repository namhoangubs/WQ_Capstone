import unittest

import numpy as np
import pandas as pd

from p1_4_mlg_transition import MLGFeatureTransformer, MLGTransitionModel, PreparedMLGDesign
from p1_10_weighted_mlg_transition import (
    GroupWeightedMLGTransitionModel,
    best_weighted_mlg_config,
    term_group,
    validate_group_weights,
)


class TestGroupWeightedMLGTransition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(17)
        rows = 240
        state = rng.integers(0, 3, size=rows)
        cls.X = pd.DataFrame({
            'STATE_LAG_0_0': (state == 0).astype(float),
            'STATE_LAG_0_1': (state == 1).astype(float),
            'STATE_LAG_0_2': (state == 2).astype(float),
            'SPXT_LAG_0': rng.normal(size=rows),
            'DURATION_STATE_0': rng.integers(0, 20, size=rows) * (state == 0),
            'DURATION_STATE_1': rng.integers(0, 20, size=rows) * (state == 1),
            'DURATION_STATE_2': rng.integers(0, 20, size=rows) * (state == 2),
        })
        signal = 0.7 * cls.X['SPXT_LAG_0'] + 0.08 * cls.X.filter(like='DURATION').sum(axis=1)
        cls.y = pd.Series((state + (signal > 0.8).astype(int)) % 3)
        transformer = MLGFeatureTransformer(n_knots=4).fit(cls.X)
        cls.prepared = PreparedMLGDesign(
            transformer=transformer,
            selected_terms=list(transformer.term_names_),
            term_p_values={term: 0.01 for term in transformer.term_names_},
            X_train=cls.X,
            y_train=cls.y,
        )

    def test_term_groups(self):
        self.assertEqual(term_group('STATE_LAG_0'), 'state')
        self.assertEqual(term_group('SPXT_LAG_0'), 'market')
        self.assertEqual(term_group('DURATION_STATE_1'), 'duration')

    def test_weights_must_be_positive_and_complete(self):
        with self.assertRaises(ValueError):
            validate_group_weights({'state': 1, 'market': 1})
        with self.assertRaises(ValueError):
            validate_group_weights({'state': 1, 'market': 1, 'duration': 0})

    def test_unit_weights_reproduce_original_mlg(self):
        original = MLGTransitionModel(self.prepared, regularization_c=0.1)
        weighted = GroupWeightedMLGTransitionModel(
            self.prepared,
            regularization_c=0.1,
            group_weights={'state': 1, 'market': 1, 'duration': 1},
        )
        np.testing.assert_allclose(
            original.predict_proba(self.X),
            weighted.predict_proba(self.X),
            atol=1e-10,
        )

    def test_column_multipliers_match_penalty_weights(self):
        weighted = GroupWeightedMLGTransitionModel(
            self.prepared,
            regularization_c=0.1,
            group_weights={'state': 1, 'market': 0.25, 'duration': 4},
        )
        slices = weighted.transformer.term_slices(self.X, weighted.selected_terms)
        for term, term_slice in slices.items():
            expected = 1 / np.sqrt(weighted.term_penalty_weights[term])
            np.testing.assert_allclose(weighted.column_multipliers_[term_slice], expected)

    def test_best_config_uses_validation_only(self):
        report = pd.DataFrame([
            {
                'MODEL': 'group_weighted_multinomial_logistic_gam',
                'SPLIT': 'validation', 'ALL_TERMS_SIGNIFICANT': True,
                'LOG_LOSS': 0.3, 'BRIER_SCORE': 0.2, 'ACCURACY': 0.8,
                'SELECTED_TERM_COUNT': 3, 'STATE_LAG': 0, 'EXOG_LAG': 0,
                'N_KNOTS': 4, 'REGULARIZATION_C': 0.1,
                'DURATION_PENALTY_WEIGHT': 4, 'MARKET_PENALTY_WEIGHT': 0.5,
            },
            {
                'MODEL': 'group_weighted_multinomial_logistic_gam',
                'SPLIT': 'prediction', 'ALL_TERMS_SIGNIFICANT': True,
                'LOG_LOSS': 0.1, 'BRIER_SCORE': 0.1, 'ACCURACY': 0.9,
                'SELECTED_TERM_COUNT': 3, 'STATE_LAG': 0, 'EXOG_LAG': 0,
                'N_KNOTS': 4, 'REGULARIZATION_C': 0.1,
                'DURATION_PENALTY_WEIGHT': 8, 'MARKET_PENALTY_WEIGHT': 0.5,
            },
        ])
        self.assertEqual(best_weighted_mlg_config(report)['SPLIT'], 'validation')


if __name__ == '__main__':
    unittest.main()
