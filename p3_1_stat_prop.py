from keras.utils import set_random_seed
from numpy import seterr, exp, cumsum, sqrt, histogram, log
from p2_0_utils import *
from statsmodels.tsa.stattools import acf
from pandas import DataFrame, concat
from matplotlib.pyplot import figure, scatter, axhline, xscale, ylim, xlabel, ylabel, gcf, close, yscale, axvline, plot
from powerlaw import Fit
from matplotlib.pyplot import legend
from pathlib import Path
from project_config import BASE_DIR, INPUT_DIR

def p3_1_stat_prop(n_states=1, n_stocks=8):
    SEED = 42
    TRAIN_DATE = '2009-06-02'
    THETA = 0.1
    MAX_VALUE = 100000
    FOLDER = BASE_DIR / 'stylized_facts'

    set_random_seed(SEED)
    seterr(divide='ignore', invalid='ignore')

    stock_px = read_csv(INPUT_DIR / 'STOCKS.csv', index_col=[0], parse_dates=[0])
    stock_rets = read_csv(BASE_DIR / f'phase_2/stock_rets_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])
    sim_rets = read_csv(BASE_DIR / f'simulations/sim_rets_q{n_states}_n{n_stocks}.csv', index_col=[0], parse_dates=[0])

    n_train = sum(stock_rets.index < TRAIN_DATE)
    real_rets = stock_rets[n_train:].drop('STATE', axis=1)
    data_size = len(real_rets)

    init_date = stock_rets.iloc[:n_train].index[-1]
    real_px = stock_px.loc[init_date, stock_rets.columns[:-1]] * exp(cumsum(real_rets))
    sim_px = stock_px.loc[init_date, stock_rets.columns[:-1]] * exp(cumsum(sim_rets))

    print('F: 0 of 10')
    auto_corrs = list()
    for rets in [real_rets, sim_rets]:
        auto_corr = list()
        for ticker in rets:
            ticker_rets = rets[f'{ticker}']
            auto_corr.append(acf(ticker_rets, nlags=data_size))

        auto_corrs.append(mean(DataFrame(auto_corr), axis=0))

    plt_auto_corrs = list()
    for auto_corr in auto_corrs:
        figure(figsize=(4, 3))
        scatter(range(data_size)[1:], auto_corr[1:], 0.5)
        axhline(lw=0.5, c='k')
        axhline(1.96 / sqrt(data_size - 1), lw=0.5, ls='--', c='k')
        axhline(-1.96 / sqrt(data_size - 1), lw=0.5, ls='--', c='k')
        axhline(lw=0.5, c='k')
        xscale('log')
        ylim([-0.25, 0.25])
        xlabel(r'Lag, $\mathit{k}$')
        ylabel('Autocorrelation')
        plt_auto_corrs.append(gcf())
        close()

    print('F:  1 of 10')
    p_law_exps_pr = list()
    bins_list = list()
    prob_rets_list = list()
    for rets in [real_rets, sim_rets]:
        p_law_exp = list()
        bins = list()
        prob_rets = list()
        for ticker in rets:
            ticker_rets = rets[f'{ticker}']
            ticker_rets_norm = (ticker_rets - mean(ticker_rets)) / std(ticker_rets)
            ticker_rets_norm = ticker_rets_norm[ticker_rets_norm > 0]
            ticker_prob_rets, ticker_bins = histogram(ticker_rets_norm, 128, density=True)

            p_law_exp.append(Fit(ticker_rets_norm, verbose=False).alpha)
            bins.append(ticker_bins)
            prob_rets.append(ticker_prob_rets / sum(ticker_prob_rets))

        p_law_exps_pr.append(round(mean(p_law_exp), 2))
        bins_list.append(mean(DataFrame(bins), axis=0))
        prob_rets_list.append(mean(DataFrame(prob_rets), axis=0))

    plt_prob_rets = list()
    for i_rets, prob_rets in iter(enumerate(prob_rets_list)):
        figure(figsize=(4, 3))
        scatter(bins_list[i_rets][:-1], prob_rets, 0.5)
        xscale('log')
        yscale('log')
        xlabel('Normalized Log Returns, r')
        ylabel(r'$\mathrm{\mathbb{P}}$(r)')
        plt_prob_rets.append(gcf())
        close()

    print('F:  2 of 10')
    vol_corrs, p_law_exps_vc = list(), list()
    for rets in [real_rets, sim_rets]:
        vol_corr, p_law_exp = list(), list()
        for ticker in rets:
            ticker_rets = rets[f'{ticker}']
            ticker_ac = acf(abs(ticker_rets), nlags=data_size)

            vol_corr.append(ticker_ac)
            p_law_exp.append(Fit(abs(ticker_ac)[:100], verbose=False).alpha)

        vol_corrs.append(mean(DataFrame(vol_corr), axis=0))
        p_law_exps_vc.append(round(mean(p_law_exp), 2))

    plt_vol_corrs = list()
    for vol_corr in vol_corrs:
        figure(figsize=(4, 3))
        scatter(range(data_size)[1:], vol_corr[1:], 0.5)
        axvline(100, lw=0.5, c='k')
        xscale('log')
        yscale('log')
        xlabel(r'Lag, $\mathit{k}$')
        ylabel('Autocorrelation')
        plt_vol_corrs.append(gcf())
        close()

    print('F:  3 of 10')
    lev_corrs = list()
    for rets in [real_rets, sim_rets]:
        lev_corr = list()
        for ticker in rets:
            ticker_rets = rets[f'{ticker}']

            ticker_lev_corr = list()
            for lag in range(100):
                ticker_rets_lag = ticker_rets.shift(-lag) ** 2
                num = mean(ticker_rets * ticker_rets_lag) - mean(ticker_rets) * mean(ticker_rets_lag)
                den = mean(ticker_rets ** 2) ** 2
                ticker_lev_corr.append(num / den)

            lev_corr.append(ticker_lev_corr)
        lev_corrs.append(mean(DataFrame(lev_corr), axis=0))

    plt_lev_corrs = list()
    for lev_corr in lev_corrs:
        figure(figsize=(4, 3))
        plot(lev_corr, lw=0.3)
        axhline(lw=0.5, c='k')
        xlabel(r'Lag, $\mathit{k}$')
        ylabel(r'$\mathit{L(k)}$')
        plt_lev_corrs.append(gcf())
        close()

    print('F:  4 of 10')
    cf_vol_corrs, d_cf_vol_corrs = list(), list()
    for rets in [real_rets, sim_rets]:
        cf_vol_corr, d_cf_vol_corr = list(), list()
        for ticker in rets:
            ticker_rets = rets[f'{ticker}']

            ticker_cf_vol_corr, ticker_d_cf_vol_corr = list(), list()
            for lag in range(-20, 21):
                ticker_cf_vol_corr.append(course_fine_corr(ticker_rets, lag))
                if lag > -1:
                    ticker_d_cf_vol_corr.append(
                        course_fine_corr(ticker_rets, lag) - course_fine_corr(ticker_rets, -lag))

            cf_vol_corr.append(ticker_cf_vol_corr)
            d_cf_vol_corr.append(ticker_d_cf_vol_corr)

        cf_vol_corrs.append(mean(DataFrame(cf_vol_corr), axis=0))
        d_cf_vol_corrs.append(mean(DataFrame(d_cf_vol_corr), axis=0))

    plt_cf_vol_corrs = list()
    for i_rets, cf_vol_corr in iter(enumerate(cf_vol_corrs)):
        figure(figsize=(4, 3))
        scatter(range(-20, 21), cf_vol_corr, 5)
        scatter(range(21), d_cf_vol_corrs[i_rets], 5, 'r')
        axhline(lw=0.5, c='k')
        legend(labels=[r'$\rho_{cf}^{\tau}$($\mathit{k}$)', r'$\Delta\rho_{cf}^{\tau}$($\mathit{k}$)'])
        ylim([-0.2, 1.0])
        xlabel(r'Lag, $\mathit{k}$')
        ylabel(r'$\rho$($\mathit{k}$)')
        plt_cf_vol_corrs.append(gcf())
        close()

    print('F:  5 of 10')
    gains_losses = list()
    for px in [real_px, sim_px]:
        gain, loss = list(), list()
        for ticker in px:
            ticker_px = px[f'{ticker}']

            ticker_gain, ticker_loss = {}, {}
            for lag in range(1, len(ticker_px)):
                px_gain = (log(ticker_px.shift(-lag)) - log(ticker_px) > THETA).values * lag
                px_loss = (log(ticker_px.shift(-lag)) - log(ticker_px) < -THETA).values * lag

                px_gain[px_gain == 0] = MAX_VALUE
                px_loss[px_loss == 0] = MAX_VALUE

                ticker_gain[lag] = px_gain
                ticker_loss[lag] = px_loss

            ticker_gain = DataFrame(ticker_gain).min(axis=1)
            ticker_loss = DataFrame(ticker_loss).min(axis=1)

            ticker_gain = ticker_gain[ticker_gain < MAX_VALUE]
            ticker_loss = ticker_loss[ticker_loss < MAX_VALUE]

            ticker_gain = DataFrame({'Gain': ticker_gain})
            ticker_loss = DataFrame({'Loss': ticker_loss})

            gain.append(ticker_gain.groupby('Gain')['Gain'].count() / len(ticker_gain))
            loss.append(ticker_loss.groupby('Loss')['Loss'].count() / len(ticker_loss))

        gain = mean(DataFrame(gain), axis=0)
        loss = mean(DataFrame(loss), axis=0)

        gain_loss = concat([gain, loss], axis=1)
        gain_loss.reset_index(inplace=True)
        gain_loss.rename(columns={'index': 'Timestep', 0: 'Gain', 1: 'Loss'}, inplace=True)
        gains_losses.append(gain_loss)

    plt_gains_losses = list()
    for gain_loss in gains_losses:
        figure(figsize=(4, 3))
        scatter(gain_loss['Timestep'], gain_loss['Gain'], 5)
        scatter(gain_loss['Timestep'], gain_loss['Loss'], 5, 'r')
        axvline(gain_loss['Timestep'][gain_loss['Gain'].idxmax()], lw=0.5)
        axvline(gain_loss['Timestep'][gain_loss['Loss'].idxmax()], lw=0.5, c='r')
        legend(labels=['Gain', 'Loss'])
        xscale('log')
        xlabel('Timestep')
        ylabel('Return Time Probability')
        plt_gains_losses.append(gcf())
        close()

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    for i_rets, n_rets in iter(enumerate(['r', 's'])):
        open(FOLDER / f'plt_{n_rets}_p_law_q{n_states}_n{n_stocks}.txt', 'w').write(
            '\n'.join(['Power Law Exponents\n',
                       f'Heavy-Tailed Distribution: {p_law_exps_pr[i_rets]}',
                       f'Volatility Clustering: {p_law_exps_vc[i_rets]}']))
        plt_auto_corrs[i_rets].savefig(FOLDER / f'plt_{n_rets}_ac_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_prob_rets[i_rets].savefig(FOLDER / f'plt_{n_rets}_pr_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_vol_corrs[i_rets].savefig(FOLDER / f'plt_{n_rets}_vc_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_lev_corrs[i_rets].savefig(FOLDER / f'plt_{n_rets}_lc_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_cf_vol_corrs[i_rets].savefig(FOLDER / f'plt_{n_rets}_cfv_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_gains_losses[i_rets].savefig(FOLDER / f'plt_{n_rets}_gl_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p3_1_stat_prop()
