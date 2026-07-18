import unittest

import numpy as np
import pandas as pd

from p1_11_pvalue_vif_mlr_transition import (
    best_pvalue_vif_mlr_config,
    build_corrected_transition_dataset,
    feature_vif_values,
    macro_vif_removal_allowed,
)


class TestPValueVIFMLRTransition(unittest.TestCase):
    def test_corrected_design_has_valid_lags_and_no_redundant_duration(self):
        index = pd.date_range('2000-01-01', periods=20, freq='D')
        states = pd.Series(([0, 0, 1, 1] * 5), index=index)
        exog = pd.DataFrame({'MACRO': np.arange(20, dtype=float)}, index=index)
        X, _, _, _, terms = build_corrected_transition_dataset(
            states, exog, max_state_lag=1, max_exog_lag=0
        )
        self.assertNotIn('DURATION', X.columns)
        self.assertNotIn('STATE_LAG_0_0', X.columns)
        self.assertGreater(X['STATE_LAG_1_1'].nunique(), 1)
        self.assertEqual(terms['STATE_LAG_1'], ['STATE_LAG_1_1'])

    def test_vif_detects_exact_dependence(self):
        x = np.arange(1, 21, dtype=float)
        design = np.column_stack([x, 2 * x])
        vifs = feature_vif_values(design, ['A', 'B'])
        self.assertTrue(np.isinf(vifs['A']))
        self.assertTrue(np.isinf(vifs['B']))

    def test_significant_macro_is_protected_from_vif_removal(self):
        self.assertFalse(macro_vif_removal_allowed('SPXT_LAG_0', 0.05))
        self.assertTrue(macro_vif_removal_allowed('SPXT_LAG_0', 0.0501))
        self.assertTrue(macro_vif_removal_allowed('DURATION_STATE_0', 0.01))

    def test_best_config_uses_validation_and_both_constraints(self):
        common = {
            'MODEL': 'pvalue_vif_selected_mlr',
            'BRIER_SCORE': 0.2,
            'ACCURACY': 0.8,
            'SELECTED_TERM_COUNT': 3,
            'SELECTED_FEATURE_COUNT': 3,
            'STATE_LAG': 0,
            'EXOG_LAG': 0,
        }
        report = pd.DataFrame([
            dict(common, SPLIT='validation', LOG_LOSS=0.3,
                 ALL_TERMS_SIGNIFICANT=True, ALL_FEATURES_WITHIN_VIF_THRESHOLD=True),
            dict(common, SPLIT='validation', LOG_LOSS=0.1,
                 ALL_TERMS_SIGNIFICANT=False, ALL_FEATURES_WITHIN_VIF_THRESHOLD=True),
            dict(common, SPLIT='prediction', LOG_LOSS=0.05,
                 ALL_TERMS_SIGNIFICANT=True, ALL_FEATURES_WITHIN_VIF_THRESHOLD=True),
        ])
        best = best_pvalue_vif_mlr_config(report)
        self.assertEqual(best['SPLIT'], 'validation')
        self.assertAlmostEqual(best['LOG_LOSS'], 0.3)


if __name__ == '__main__':
    unittest.main()
