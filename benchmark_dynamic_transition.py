from pandas import concat

from p1_4_dynamic_transition import TRAIN_END, VALIDATION_END, save_dynamic_transition_report
from p1_4_mlg_transition import best_mlg_config, save_mlg_transition_report
from project_config import BASE_DIR


def main():
    print(f'Train window (both models): start of data to {TRAIN_END}')
    print(f'Validation window (lag selection only): {TRAIN_END} to {VALIDATION_END}')
    print(f'Prediction window (final comparison): {VALIDATION_END} onward')

    for n_states in [2, 4]:
        report, output_path = save_dynamic_transition_report(n_states=n_states)

        validation_rows = report[(report.MODEL == 'dynamic_duration_logit') & (report.SPLIT == 'validation')]
        best_dynamic = validation_rows.sort_values('LOG_LOSS').head(1)
        state_lag = int(best_dynamic.STATE_LAG.iloc[0])
        exog_lag = int(best_dynamic.EXOG_LAG.iloc[0])

        selected = report[(report.STATE_LAG == state_lag) & (report.EXOG_LAG == exog_lag)]

        print(f'\nN_STATES={n_states}')
        print(f'Report saved to: {output_path}')
        print(f'Best lag config by validation log loss: state_lag={state_lag}, exog_lag={exog_lag}')
        print('Fixed versus dynamic at that config (prediction split is the out-of-sample comparison):')
        print(selected.to_string(index=False))

        mlg_report, mlg_output_path = save_mlg_transition_report(n_states=n_states)
        best_mlg = best_mlg_config(mlg_report)
        mlg_selected = mlg_report[
            (mlg_report.STATE_LAG == best_mlg.STATE_LAG)
            & (mlg_report.EXOG_LAG == best_mlg.EXOG_LAG)
            & (mlg_report.N_KNOTS == best_mlg.N_KNOTS)
            & (mlg_report.REGULARIZATION_C == best_mlg.REGULARIZATION_C)
        ]

        comparison = concat([
            selected[selected.MODEL == 'fixed_hmm_transition'],
            selected[selected.MODEL == 'dynamic_duration_logit'],
            mlg_selected,
        ], ignore_index=True, sort=False)
        comparison_path = BASE_DIR / 'phase_1' / f'transition_model_comparison_q{n_states}.csv'
        comparison.to_csv(comparison_path, index=False)

        print(f'MLG report saved to: {mlg_output_path}')
        print(
            'Best MLG config by validation log loss/Brier score with all retained terms '
            f'significant at 5%: state_lag={int(best_mlg.STATE_LAG)}, '
            f'exog_lag={int(best_mlg.EXOG_LAG)}, knots={int(best_mlg.N_KNOTS)}, '
            f'C={float(best_mlg.REGULARIZATION_C):g}'
        )
        print('Fixed versus MLR versus MLG (prediction split is the final out-of-sample comparison):')
        print(comparison.to_string(index=False))
        print(f'Three-model comparison saved to: {comparison_path}')


if __name__ == '__main__':
    main()
