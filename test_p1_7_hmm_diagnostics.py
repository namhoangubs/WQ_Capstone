import unittest
import os

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
from hmmlearn.hmm import GaussianHMM

from p1_7_hmm_diagnostics import (
    covariance_diagnostic_rows,
    duration_geometric_rows,
    emission_normality_rows,
    filtered_state_probability,
    fit_multistart_hmms,
    goodness_of_fit_row,
    one_step_probability_integral_transforms,
    regime_stability_rows,
    residual_diagnostic_rows,
    transition_homogeneity_test,
)


class TestHMMDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = GaussianHMM(n_components=2, covariance_type='full', random_state=5)
        generator.startprob_ = np.array([0.7, 0.3])
        generator.transmat_ = np.array([[0.96, 0.04], [0.08, 0.92]])
        generator.means_ = np.array([[-0.8, 0.2], [1.0, -0.3]])
        generator.covars_ = np.array([np.eye(2) * 0.25, np.eye(2) * 0.35])
        sample, _ = generator.sample(900)
        cls.train = sample[:600]
        cls.validation = sample[600:750]
        cls.prediction = sample[750:]

    def test_q2_q3_q4_fit_and_filtered_pits(self):
        models, candidates = fit_multistart_hmms(
            self.train,
            n_starts=2,
            random_seed=17,
            max_iter=300,
        )
        self.assertEqual(sorted(models), [2, 3, 4])
        self.assertEqual(len(candidates), 6)

        gof = goodness_of_fit_row(
            2,
            models[2],
            self.train,
            self.validation,
            self.prediction,
        )
        self.assertTrue(np.isfinite(gof['BIC']))
        self.assertTrue(np.isfinite(gof['VALIDATION_LOG_LIKELIHOOD_PER_ROW']))

        initial = filtered_state_probability(models[2], self.train)
        pits, final_probability = one_step_probability_integral_transforms(
            models[2],
            self.validation,
            initial,
        )
        self.assertEqual(pits.shape, self.validation.shape)
        self.assertTrue(np.all((pits > 0) & (pits < 1)))
        self.assertAlmostEqual(float(final_probability.sum()), 1.0)

        stability = regime_stability_rows(
            2,
            models[2],
            self.train,
            ['SPXT', 'LT11TRUU'],
        )
        self.assertEqual(len(stability), 2)
        self.assertTrue(all('UNSTABLE_FLAG' in row for row in stability))

    def test_transition_homogeneity_detects_change(self):
        stable = np.array([[90, 10], [180, 20], [45, 5]])
        changing = np.array([[95, 5], [50, 50], [90, 10]])
        _, stable_p_value, _ = transition_homogeneity_test(stable)
        _, changing_p_value, _ = transition_homogeneity_test(changing)
        self.assertGreater(stable_p_value, 0.05)
        self.assertLess(changing_p_value, 0.05)

    def test_covariance_condition_number_flags_near_singular_regime(self):
        model = GaussianHMM(n_components=2, covariance_type='full')
        model.n_features = 2
        model.means_ = np.array([[-1.0, 0.0], [1.0, 0.0]])
        model.covars_ = np.array([
            np.eye(2),
            np.diag([1.0, 1e-12]),
        ])

        rows = covariance_diagnostic_rows(
            2,
            model,
            ['SPXT', 'BOND'],
            condition_threshold=1e8,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]['NUMERICALLY_STABLE'])
        self.assertFalse(rows[1]['NUMERICALLY_STABLE'])
        self.assertGreater(rows[1]['COVARIANCE_CONDITION_NUMBER'], 1e8)

    def test_additive_distribution_and_volatility_diagnostics(self):
        rng = np.random.default_rng(31)
        states = np.repeat([0, 1], 250)
        data = np.column_stack([
            rng.normal(size=500),
            np.r_[rng.normal(size=250), rng.standard_t(df=3, size=250)],
        ])

        class PredictableModel:
            means_ = np.array([[-1.0, 0.0], [1.0, 0.0]])
            transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])

            @staticmethod
            def predict(values):
                return states[:len(values)]

        normality = emission_normality_rows(
            2,
            PredictableModel(),
            data,
            ['SPXT', 'BOND'],
        )
        residuals = residual_diagnostic_rows(
            2,
            'validation',
            rng.uniform(size=(400, 2)),
            ['SPXT', 'BOND'],
        )

        self.assertEqual(len(normality), 4)
        self.assertTrue(all(np.isfinite(row['JARQUE_BERA_P_VALUE']) for row in normality))
        self.assertEqual(len(residuals), 2)
        self.assertTrue(all('ARCH_LM_P_VALUE' in row for row in residuals))
        self.assertTrue(all('SQUARED_PIT_LJUNG_BOX_LAG_20_P_VALUE' in row for row in residuals))

    def test_duration_bootstrap_reports_hmm_geometric_assumption(self):
        rng = np.random.default_rng(41)
        states = []
        state = 0
        while len(states) < 500:
            states.extend([state] * int(rng.geometric(0.15 if state == 0 else 0.25)))
            state = 1 - state
        states = np.asarray(states[:500])
        data = rng.normal(size=(500, 2))

        class DurationModel:
            means_ = np.array([[-1.0, 0.0], [1.0, 0.0]])
            transmat_ = np.array([[0.85, 0.15], [0.25, 0.75]])

            @staticmethod
            def predict(values):
                return states[:len(values)]

        rows = duration_geometric_rows(
            2,
            DurationModel(),
            data,
            ['SPXT', 'BOND'],
            n_bootstrap=50,
            random_seed=41,
        )

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['COMPLETE_SPELL_COUNT'] >= 10 for row in rows))
        self.assertTrue(all(np.isfinite(row['BOOTSTRAP_P_VALUE']) for row in rows))


if __name__ == '__main__':
    unittest.main()
