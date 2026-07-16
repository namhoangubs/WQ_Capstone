from p1_4_dynamic_transition import TRAIN_END, VALIDATION_END, save_dynamic_transition_report


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


if __name__ == '__main__':
    main()
