from p1_4_dynamic_transition import save_dynamic_transition_report


def main():
    for n_states in [2, 4]:
        report, output_path = save_dynamic_transition_report(n_states=n_states)
        best_dynamic = report[report.MODEL == 'dynamic_duration_logit'].sort_values('LOG_LOSS').head(1)
        comparable_fixed = report[
            (report.MODEL == 'fixed_hmm_transition')
            & (report.STATE_LAG == int(best_dynamic.STATE_LAG.iloc[0]))
            & (report.EXOG_LAG == int(best_dynamic.EXOG_LAG.iloc[0]))
        ]

        print(f'\nN_STATES={n_states}')
        print(f'Report saved to: {output_path}')
        print('Best dynamic transition model:')
        print(best_dynamic.to_string(index=False))
        print('Fixed transition baseline on same validation rows:')
        print(comparable_fixed.to_string(index=False))


if __name__ == '__main__':
    main()
