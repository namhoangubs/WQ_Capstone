from numpy.random import default_rng
from pandas import read_csv, DataFrame
from itertools import product
from hmmlearn.hmm import GaussianHMM
from numpy import unique
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p1_0_hmm_params():
    SEED = 42
    N_SEEDS = 10000
    TRAIN_DATE = '2009-06-02'
    # https://fred.stlouisfed.org/series/GDPC1
    # https://fredhelp.stlouisfed.org/fred/data/understanding-the-data/recession-bars/
    FOLDER = BASE_DIR / 'phase_1'

    seeds = sorted(default_rng(seed=SEED).choice(range(1000000), N_SEEDS, False, shuffle=False))
    exog_rets = read_csv(BASE_DIR / 'processed' / 'exog_rets.csv', index_col=[0], parse_dates=[0])

    n_train = sum(exog_rets.index < TRAIN_DATE)
    train_data_2 = exog_rets[:n_train][['SPXT', 'LT11TRUU']]
    train_data_4 = exog_rets[:n_train]

    params = list()
    for n_states, seed in product([2, 4], seeds):
        try:
            train_data = train_data_2 if n_states == 2 else train_data_4
            hmm = GaussianHMM(n_states, 'full', random_state=seed, n_iter=10000, tol=1e-5, implementation='scaling')
            hmm.fit(train_data)
            q_count = unique(hmm.predict(train_data), return_counts=True)[-1]

            params.append({
                'N_STATES': n_states,
                'SEED': seed,
                'LL': round(hmm.score(train_data), 2),
                'AIC': round(hmm.aic(train_data), 2),
                'BIC': round(hmm.bic(train_data), 2),
                'MIN_COUNT': min(q_count),
                'CONVERGED': hmm.monitor_.history[-1] - hmm.monitor_.history[-2] > 0,
                'VALID': len(q_count) == n_states})

            print(f'Q: {n_states} | S: {seed:6d}')

        except ValueError:
            pass

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    DataFrame(params).to_csv(FOLDER / 'hmm_params.csv', index=False)


if __name__ == '__main__':
    p1_0_hmm_params()