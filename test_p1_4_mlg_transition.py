import unittest

import numpy as np
import pandas as pd

from p1_4_mlg_transition import (
    MLGFeatureTransformer,
    MLGTransitionModel,
    PreparedMLGDesign,
    select_significant_terms,
)


class TestMLGTransition(unittest.TestCase):
    def test_state_specific_duration_terms_replace_redundant_global_duration(self):
        rng = np.random.default_rng(21)
        row_count = 300
        state = rng.integers(0, 2, row_count)
        duration = np.where(
            state == 0,
            rng.integers(1, 20, row_count),
            rng.integers(1, 12, row_count),
        )
        X = pd.DataFrame({
            'STATE_LAG_0_0': state == 0,
            'STATE_LAG_0_1': state == 1,
            'EXOGENOUS_SIGNAL': rng.normal(size=row_count),
            'DURATION': duration,
            'DURATION_STATE_0': duration * (state == 0),
            'DURATION_STATE_1': duration * (state == 1),
        }).astype(float)

        transformer = MLGFeatureTransformer(n_knots=4).fit(X)
        design = transformer.transform(X, penalized=False)
        design_with_intercept = np.column_stack([np.ones(row_count), design])

        self.assertNotIn('DURATION', transformer.term_names_)
        self.assertIn('DURATION_STATE_0', transformer.term_names_)
        self.assertIn('DURATION_STATE_1', transformer.term_names_)
        self.assertEqual(
            np.linalg.matrix_rank(design_with_intercept),
            design_with_intercept.shape[1],
        )

    def test_joint_screening_and_regularized_probabilities(self):
        rng = np.random.default_rng(7)
        row_count = 700
        state = rng.integers(0, 3, row_count)
        signal = rng.normal(size=row_count)
        noise = rng.normal(size=row_count)
        logits = np.column_stack([
            np.zeros(row_count),
            1.2 * (state == 1) - 0.8 * (state == 2) + 1.4 * np.sin(signal),
            -0.9 * (state == 1) + 1.0 * (state == 2) - 1.1 * signal ** 2,
        ])
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        X = pd.DataFrame({
            'STATE_LAG_0_0': state == 0,
            'STATE_LAG_0_1': state == 1,
            'STATE_LAG_0_2': state == 2,
            'SIGNAL': signal,
            'NOISE': noise,
        }).astype(float)
        y = pd.Series([rng.choice(3, p=row) for row in probabilities])

        transformer = MLGFeatureTransformer(n_knots=4).fit(X)
        selected, p_values, method, _, _ = select_significant_terms(transformer, X, y)

        self.assertIn('STATE_LAG_0', selected)
        self.assertIn('SIGNAL', selected)
        self.assertNotIn('NOISE', selected)
        self.assertEqual(method, 'wald')
        self.assertTrue(all(p_values[term] <= 0.05 for term in selected))

        prepared = PreparedMLGDesign(transformer, selected, p_values, X, y)
        model = MLGTransitionModel(prepared, regularization_c=1.0)
        predicted = model.predict_proba(X.iloc[:10])
        self.assertEqual(predicted.shape, (10, 3))
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0)

        effects, contrasts = model.term_log_odds(X.iloc[:10], 'SIGNAL')
        self.assertEqual(effects.shape, (10, 3))
        self.assertEqual(contrasts, ['1 vs 0', '2 vs 0', '2 vs 1'])

    def test_state_duration_partial_effects_use_focal_state_contrasts(self):
        rng = np.random.default_rng(13)
        row_count = 240
        current_state = rng.integers(0, 3, row_count)
        duration = rng.integers(1, 40, row_count)
        X = pd.DataFrame({
            'DURATION': duration,
            'DURATION_STATE_0': duration * (current_state == 0),
            'DURATION_STATE_1': duration * (current_state == 1),
            'DURATION_STATE_2': duration * (current_state == 2),
        }).astype(float)
        y = pd.Series(np.tile([0, 1, 2], row_count // 3 + 1)[:row_count])

        transformer = MLGFeatureTransformer(n_knots=4).fit(X)
        prepared = PreparedMLGDesign(
            transformer,
            ['DURATION_STATE_1'],
            {'DURATION_STATE_1': 0.01},
            X,
            y,
        )
        model = MLGTransitionModel(prepared, regularization_c=1.0)

        effects, contrasts = model.term_log_odds(X.iloc[:8], 'DURATION_STATE_1')
        self.assertEqual(effects.shape, (8, 2))
        self.assertEqual(contrasts, ['1 vs 0', '1 vs 2'])

    def test_regularized_bootstrap_screening_selects_signal(self):
        rng = np.random.default_rng(9)
        row_count = 260
        signal = rng.normal(size=row_count)
        noise = rng.normal(size=row_count)
        logits = np.column_stack([
            np.zeros(row_count),
            2.0 * signal,
            -1.8 * signal,
        ])
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        X = pd.DataFrame({
            'SIGNAL': signal,
            'NOISE': noise,
        }).astype(float)
        y = pd.Series([rng.choice(3, p=row) for row in probabilities])

        transformer = MLGFeatureTransformer(n_knots=4).fit(X)
        selected, p_values, method, improvements, replicates = select_significant_terms(
            transformer,
            X,
            y,
            alpha=0.15,
            inference_method='regularized_bootstrap',
            n_bootstrap=19,
            random_state=19,
        )

        self.assertEqual(method, 'regularized_bootstrap')
        self.assertIn('SIGNAL', selected)
        self.assertNotIn('NOISE', selected)
        self.assertLessEqual(p_values['SIGNAL'], 0.15)
        self.assertGreater(improvements['SIGNAL'], 0.0)
        self.assertGreaterEqual(replicates['SIGNAL'], 1)


if __name__ == '__main__':
    unittest.main()
