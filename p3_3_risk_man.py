from os import environ
from keras.utils import set_random_seed
from p2_0_utils import *
from scipy.stats import multivariate_normal
from numpy import mean
from tensorflow import random
from pandas import DataFrame
from functools import reduce
from matplotlib.pyplot import figure, scatter, axhline, legend, ylim, xlabel, ylabel, locator_params, gcf, close, plot
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p3_3_risk_man(n_states=1, n_stocks=8):
    SEED = 42
    TRAIN_DATE = '2009-06-02'
    Z_DIM = 128
    WINDOW_SIZE = 256
    N_SIMS = 10000
    EXCEED_SIZE = 250
    FOLDER = BASE_DIR / 'risk_management'

    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_random_seed(SEED)

    stock_rets = read_csv(BASE_DIR / f'phase_2/stock_rets_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])
    n_train = sum(stock_rets.index < TRAIN_DATE)
    windows = range(len(stock_rets[n_train:]))

    real_rets = stock_rets[n_train:]
    real_pf = real_rets.drop('STATE', axis=1).mean(axis=1)
    real_pf.name = 'REAL_PF'

    sims = read_csv(BASE_DIR / f'simulations/sims_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])
    g_net = generator(Z_DIM, n_stocks)

    v_norm_90, v_norm_95, v_norm_975, v_norm_99 = list(), list(), list(), list()
    c_norm_90, c_norm_95, c_norm_975, c_norm_99 = list(), list(), list(), list()
    v_wgan_90, v_wgan_95, v_wgan_975, v_wgan_99 = list(), list(), list(), list()
    c_wgan_90, c_wgan_95, c_wgan_975, c_wgan_99 = list(), list(), list(), list()
    for window in windows:
        window_data = window_load(stock_rets, window, n_train, WINDOW_SIZE).drop('STATE', axis=1)
        mv_norm = multivariate_normal(window_data.mean(), window_data.cov(), allow_singular=True)
        sim_n_pfs = sorted(mv_norm.rvs(N_SIMS, random_state=SEED).mean(axis=1))

        v_norm_90.append(sim_n_pfs[int((1 - 0.9) * N_SIMS)])
        v_norm_95.append(sim_n_pfs[int((1 - 0.95) * N_SIMS)])
        v_norm_975.append(sim_n_pfs[int((1 - 0.975) * N_SIMS)])
        v_norm_99.append(sim_n_pfs[int((1 - 0.99) * N_SIMS)])

        c_norm_90.append(mean(sim_n_pfs[:int((1 - 0.9) * N_SIMS) + 1]))
        c_norm_95.append(mean(sim_n_pfs[:int((1 - 0.95) * N_SIMS) + 1]))
        c_norm_975.append(mean(sim_n_pfs[:int((1 - 0.975) * N_SIMS) + 1]))
        c_norm_99.append(mean(sim_n_pfs[:int((1 - 0.99) * N_SIMS) + 1]))

        states = unique(sims.iloc[window, :], return_counts=True)
        sim_w_pfs = list()
        for state, batch_size in zip(states[0], states[1]):
            g_net.load_weights(BASE_DIR / 'models' / f'w{window}_s{state}_q{n_states}.h5')
            z = random.normal((batch_size, Z_DIM))
            sim_w_pfs = sim_w_pfs + reduce_mean(g_net(z), axis=1).numpy().tolist()
        sim_w_pfs = sorted(sim_w_pfs)

        v_wgan_90.append(sim_w_pfs[int((1 - 0.9) * N_SIMS)])
        v_wgan_95.append(sim_w_pfs[int((1 - 0.95) * N_SIMS)])
        v_wgan_975.append(sim_w_pfs[int((1 - 0.975) * N_SIMS)])
        v_wgan_99.append(sim_w_pfs[int((1 - 0.99) * N_SIMS)])

        c_wgan_90.append(mean(sim_w_pfs[:int((1 - 0.9) * N_SIMS) + 1]))
        c_wgan_95.append(mean(sim_w_pfs[:int((1 - 0.95) * N_SIMS) + 1]))
        c_wgan_975.append(mean(sim_w_pfs[:int((1 - 0.975) * N_SIMS) + 1]))
        c_wgan_99.append(mean(sim_w_pfs[:int((1 - 0.99) * N_SIMS) + 1]))

        print(f'W: {window:4d} of {windows[-1]}')

    var_list = [real_pf,
                DataFrame(v_norm_90, real_pf.index, columns=['V_NORM_90']),
                DataFrame(v_norm_95, real_pf.index, columns=['V_NORM_95']),
                DataFrame(v_norm_975, real_pf.index, columns=['V_NORM_975']),
                DataFrame(v_norm_99, real_pf.index, columns=['V_NORM_99']),
                DataFrame(c_norm_90, real_pf.index, columns=['C_NORM_90']),
                DataFrame(c_norm_95, real_pf.index, columns=['C_NORM_95']),
                DataFrame(c_norm_975, real_pf.index, columns=['C_NORM_975']),
                DataFrame(c_norm_99, real_pf.index, columns=['C_NORM_99']),
                DataFrame(v_wgan_90, real_pf.index, columns=['V_WGAN_90']),
                DataFrame(v_wgan_95, real_pf.index, columns=['V_WGAN_95']),
                DataFrame(v_wgan_975, real_pf.index, columns=['V_WGAN_975']),
                DataFrame(v_wgan_99, real_pf.index, columns=['V_WGAN_99']),
                DataFrame(c_wgan_90, real_pf.index, columns=['C_WGAN_90']),
                DataFrame(c_wgan_95, real_pf.index, columns=['C_WGAN_95']),
                DataFrame(c_wgan_975, real_pf.index, columns=['C_WGAN_975']),
                DataFrame(c_wgan_99, real_pf.index, columns=['C_WGAN_99'])]
    var_rets = reduce(lambda left, right: merge(left, right, on='DATE'), var_list)

    exceed_windows = range(EXCEED_SIZE, len(stock_rets[n_train:]))
    exceed_n_975, exceed_n_99 = list(), list()
    exceed_w_975, exceed_w_99 = list(), list()
    for window in exceed_windows:
        exceed_n_975.append(sum((real_pf < var_rets.V_NORM_975)[(window - EXCEED_SIZE):window]))
        exceed_w_975.append(sum((real_pf < var_rets.V_WGAN_975)[(window - EXCEED_SIZE):window]))
        exceed_n_99.append(sum((real_pf < var_rets.V_NORM_99)[(window - EXCEED_SIZE):window]))
        exceed_w_99.append(sum((real_pf < var_rets.V_WGAN_99)[(window - EXCEED_SIZE):window]))

    plt_model = 'WGAN' if n_states == 1 else 'HMM-WGAN'

    figure(figsize=(6, 3))
    scatter(exceed_windows, exceed_n_975, 5, 'k')
    scatter(exceed_windows, exceed_w_975, 5)
    axhline(30, lw=0.5, ls='--', c='r')
    legend(['97.5%-VaR (Normal)', f'97.5%-VaR ({plt_model})', 'Exceedance Limit'],
           prop={'size': 5})
    ylim(0, max(max(exceed_n_975), max(exceed_w_975), 30) + 10)
    xlabel('Window')
    ylabel('Exceedance')
    locator_params(axis='y', integer=True)
    plt_exceed_975 = gcf()
    close()

    figure(figsize=(6, 3))
    scatter(exceed_windows, exceed_n_99, 5, 'k')
    scatter(exceed_windows, exceed_w_99, 5)
    axhline(12, lw=0.5, ls='--', c='r')
    legend(['99%-VaR (Normal)', f'99%-VaR ({plt_model})', 'Exceedance Limit'],
           prop={'size': 5})
    ylim(0, max(max(exceed_n_99), max(exceed_w_99), 12) + 10)
    xlabel('Window')
    ylabel('Exceedance')
    locator_params(axis='y', integer=True)
    plt_exceed_99 = gcf()
    close()

    figure(figsize=(6, 3))
    plot(real_pf, lw=0.3)
    plot(var_rets.V_NORM_90, lw=0.3, ls='-.', c='g')
    plot(var_rets.V_NORM_95, lw=0.3, ls='-.', c='y')
    plot(var_rets.V_NORM_99, lw=0.3, ls='-.', c='r')
    legend(['Actual Returns', '90%-VaR (Normal)', '95%-VaR (Normal)', '99%-VaR (Normal)'],
           prop={'size': 5})
    axhline(lw=0.5, c='k')
    xlabel('Time')
    ylabel('Returns')
    plt_var_n = gcf()
    close()

    figure(figsize=(6, 3))
    plot(real_pf, lw=0.3)
    plot(var_rets.V_WGAN_90, lw=0.3, c='g')
    plot(var_rets.V_WGAN_95, lw=0.3, c='y')
    plot(var_rets.V_WGAN_99, lw=0.3, c='r')
    legend(['Actual Returns', f'90%-VaR ({plt_model})', f'95%-VaR ({plt_model})', f'99%-VaR ({plt_model})'],
           prop={'size': 5})
    axhline(lw=0.5, c='k')
    xlabel('Time')
    ylabel('Returns')
    plt_var_w = gcf()
    close()

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    var_rets.to_csv(FOLDER / f'var_q{n_states}_n{n_stocks}.csv')
    round(DataFrame([[mean(real_pf < v_norm_90), mean(real_pf < v_wgan_90)],
                     [mean(real_pf < v_norm_95), mean(real_pf < v_wgan_95)],
                     [mean(real_pf < v_norm_99), mean(real_pf < v_wgan_99)]],
                    columns=['Normal', f'{plt_model}'],
                    index=['90%', '95%', '99%']), 5).to_csv(FOLDER / f'var_e_q{n_states}_n{n_stocks}.csv')
    round(DataFrame([[mean((var_rets.V_NORM_90 - real_pf)[real_pf < v_norm_90]),
                      mean((var_rets.V_WGAN_90 - real_pf)[real_pf < v_wgan_90])],
                     [mean((var_rets.V_NORM_95 - real_pf)[real_pf < v_norm_95]),
                      mean((var_rets.V_WGAN_95 - real_pf)[real_pf < v_wgan_95])],
                     [mean((var_rets.V_NORM_99 - real_pf)[real_pf < v_norm_99]),
                      mean((var_rets.V_WGAN_99 - real_pf)[real_pf < v_wgan_99])]],
                    columns=['Normal', f'{plt_model}'],
                    index=['90%', '95%', '99%']), 5).to_csv(FOLDER / f'var_e_delta_q{n_states}_n{n_stocks}.csv')
    round(DataFrame([[mean((real_pf - var_rets.V_NORM_90)[real_pf > v_norm_90]),
                      mean((real_pf - var_rets.V_WGAN_90)[real_pf > v_wgan_90])],
                     [mean((real_pf - var_rets.V_NORM_95)[real_pf > v_norm_95]),
                      mean((real_pf - var_rets.V_WGAN_95)[real_pf > v_wgan_95])],
                     [mean((real_pf - var_rets.V_NORM_99)[real_pf > v_norm_99]),
                      mean((real_pf - var_rets.V_WGAN_99)[real_pf > v_wgan_99])]],
                    columns=['Normal', f'{plt_model}'],
                    index=['90%', '95%', '99%']), 5).to_csv(FOLDER / f'var_c_delta_q{n_states}_n{n_stocks}.csv')
    plt_exceed_975.savefig(FOLDER / f'plt_var_e_975_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
    plt_exceed_99.savefig(FOLDER / f'plt_var_e_99_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
    plt_var_n.savefig(FOLDER / f'plt_var_n_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
    plt_var_w.savefig(FOLDER / f'plt_var_w_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p3_3_risk_man()