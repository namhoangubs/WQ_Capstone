from pandas import read_csv, concat, DataFrame
from numpy import unique
from pathlib import Path
from project_config import BASE_DIR, INPUT_DIR

def p1_1_best_params():
    FOLDER = BASE_DIR / 'phase_1'

    params = read_csv(BASE_DIR / 'phase_1' / 'hmm_params.csv')
    params = params[params.CONVERGED & params.VALID]

    params_m_ll = list()
    for n_states in unique(params.N_STATES):
        params_n_s = params[params.N_STATES == n_states]
        params_m_c = params_n_s[params_n_s.MIN_COUNT > 9]

        if params_m_c.empty:
            print(f'No valid HMM parameter set found for N_STATES={n_states} with MIN_COUNT > 9')
            continue

        best_param = params_m_c.sort_values(
            ['LL', 'MIN_COUNT', 'BIC', 'AIC', 'SEED'],
            ascending=[False, False, True, True, True]
        ).head(1)
        params_m_ll.append(best_param)

    if not params_m_ll:
        raise ValueError('No HMM parameter sets passed CONVERGED, VALID, and MIN_COUNT > 9 filters.')

    best_params = concat(params_m_ll)[['N_STATES', 'SEED', 'MIN_COUNT']]

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    DataFrame(best_params).to_csv(FOLDER / 'best_params.csv', index=False)


if __name__ == '__main__':
    p1_1_best_params()
