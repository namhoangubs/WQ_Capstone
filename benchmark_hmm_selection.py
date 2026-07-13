from time import perf_counter

from numpy import unique
from pandas import concat, read_csv

from project_config import BASE_DIR


def old_select(params):
    selected = []
    for n_states in unique(params.N_STATES):
        params_n_s = params[params.N_STATES == n_states]

        exp = 0
        guard = 0
        while len(params) > 0:
            guard += 1
            if guard > 1000:
                raise RuntimeError(f'Old selector did not terminate for N_STATES={n_states}')

            bound = (10 ** exp - 1) / 10 ** exp
            params_b = params_n_s[params_n_s.LL > bound * params_n_s.LL.max()]

            if len(params_b[params_b.MIN_COUNT > 9]) < 1:
                exp -= 1
                bound = (10 ** exp - 1) / 10 ** exp
                params_n_s = params_n_s[params_n_s.LL > bound * params_n_s.LL.max()]
                break

            exp += 1

        params_m_c = params_n_s[params_n_s.MIN_COUNT > 9]
        selected.append(params_m_c[params_m_c.LL == params_m_c.LL.max()])

    return concat(selected)[['N_STATES', 'SEED', 'MIN_COUNT']]


def new_select(params):
    selected = []
    for n_states in unique(params.N_STATES):
        params_n_s = params[params.N_STATES == n_states]
        params_m_c = params_n_s[params_n_s.MIN_COUNT > 9]

        if params_m_c.empty:
            continue

        selected.append(params_m_c.sort_values(
            ['LL', 'MIN_COUNT', 'BIC', 'AIC', 'SEED'],
            ascending=[False, False, True, True, True]
        ).head(1))

    if not selected:
        raise ValueError('No HMM parameter sets passed CONVERGED, VALID, and MIN_COUNT > 9 filters.')

    return concat(selected)[['N_STATES', 'SEED', 'MIN_COUNT']]


def timed(label, fn, params):
    start = perf_counter()
    result = fn(params)
    seconds = perf_counter() - start
    print(f'{label} rows: {len(result)} time_seconds={seconds:.6f}')
    return result


def main():
    params = read_csv(BASE_DIR / 'phase_1' / 'hmm_params.csv')
    params = params[params.CONVERGED & params.VALID]

    print(f'Filtered HMM candidate rows: {len(params)}')
    old_best = timed('Old selector', old_select, params)
    print(old_best.groupby('N_STATES').size().to_string())

    print()
    new_best = timed('New selector', new_select, params)
    print(new_best.to_string(index=False))


if __name__ == '__main__':
    main()
