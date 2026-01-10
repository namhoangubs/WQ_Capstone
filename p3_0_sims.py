from os import environ
from keras.utils import set_random_seed
from p2_0_utils import *
from numpy.random import multinomial
from numpy import argmax
from pandas import DataFrame
from numpy.random import default_rng
from tensorflow import random
from matplotlib.pyplot import figure, plot, axhline, xlabel, ylabel, gcf, close
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p3_0_sims(n_states=1, n_stocks=8):
    SEED = 42
    TRAIN_DATE = '2009-06-02'
    Z_DIM = 128
    N_SIMS = 10000
    FOLDER = BASE_DIR / 'simulations'

    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_random_seed(SEED)

    stock_rets = read_csv(BASE_DIR / f'phase_2/stock_rets_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])
    n_train = sum(stock_rets.index < TRAIN_DATE)
    windows = range(len(stock_rets[n_train:]))

    A = read_csv(BASE_DIR / f'phase_1/hmm_trans_mat_q{n_states}.csv', index_col=[0])
    g_net = generator(Z_DIM, n_stocks)

    window_sims = list()
    for window in windows:
        q = int(stock_rets.iloc[n_train + window - 1].STATE)
        sims = multinomial(1, A.iloc[q], N_SIMS)
        window_sims.append([argmax(sim) for sim in sims])

        print(f'W: {window:4d} of {windows[-1]}')

    sims = DataFrame(window_sims, stock_rets[n_train:].index)
    sim = default_rng(seed=SEED).choice(range(N_SIMS), 1)[0]

    sim_rets = list()
    for window, state in zip(range(len(sims[sim])), sims[sim]):
        g_net.load_weights(BASE_DIR / 'models' / f'w{window}_s{state}_q{n_states}.weights.h5')
        z = random.normal((1, Z_DIM))
        sim_rets.append(g_net(z).numpy().tolist()[0])

        print(f'W: {window:4d} of {windows[-1]}')

    sim_rets = DataFrame(sim_rets, columns=stock_rets.columns[:-1], index=stock_rets[n_train:].index)

    figure(figsize=(6, 3))
    plot(sim_rets.mean(axis=1), lw=0.3)
    axhline(lw=0.5, c='k')
    xlabel('Time')
    ylabel('Returns')
    plt_sim_rets = gcf()
    close()

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    sims.to_csv(FOLDER / f'sims_q{n_states}_n{n_stocks}.csv')
    sim_rets.to_csv(FOLDER / f'sim_rets_q{n_states}_n{n_stocks}.csv')
    plt_sim_rets.savefig(FOLDER / f'plt_sim_rets_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p3_0_sims()