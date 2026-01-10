from pandas import read_csv, concat, DataFrame
from numpy import unique
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p1_1_best_params():
    FOLDER = BASE_DIR / 'phase_1'

    params = read_csv(BASE_DIR / 'phase_1' / 'hmm_params.csv')
    params = params[params.CONVERGED & params.VALID]

    params_m_ll = list()
    for n_states in unique(params.N_STATES):
        params_n_s = params[params.N_STATES == n_states]

        exp = 0
        while len(params) > 0:
            bound = (10 ** exp - 1) / 10 ** exp
            params_b = params_n_s[params_n_s.LL > bound * params_n_s.LL.max()]

            if len(params_b[params_b.MIN_COUNT > 9]) < 1:
                exp -= 1
                bound = (10 ** exp - 1) / 10 ** exp
                params_n_s = params_n_s[params_n_s.LL > bound * params_n_s.LL.max()]
                break
            else:
                exp += 1

        params_m_c = params_n_s[params_n_s.MIN_COUNT > 9]
        params_m_ll.append(params_m_c[params_m_c.LL == params_m_c.LL.max()])

    best_params = concat(params_m_ll)[['N_STATES', 'SEED', 'MIN_COUNT']]

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    DataFrame(best_params).to_csv(FOLDER / 'best_params.csv', index=False)


if __name__ == '__main__':
    p1_1_best_params()