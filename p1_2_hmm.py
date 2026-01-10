from pandas import read_csv, DataFrame, concat
from hmmlearn.hmm import GaussianHMM
from matplotlib.pyplot import figure, scatter, axvspan, axhline, legend, xlabel, ylabel, gcf, close
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p1_2_hmm():
    TRAIN_DATE = '2009-06-02'
    WINDOW_SIZE = 256
    FOLDER = BASE_DIR / 'phase_1'

    best_params = read_csv(BASE_DIR / 'phase_1' / 'best_params.csv')
    exog_data = read_csv(BASE_DIR / 'processed' / 'exog_rets.csv', index_col=[0], parse_dates=[0])
    stock_rets = read_csv(BASE_DIR / 'processed' / 'stock_rets.csv', index_col=[0], parse_dates=[0])
    exog_px = read_csv(BASE_DIR / 'processed' / 'exog_px.csv', index_col=[0], parse_dates=[0])

    for n_states in [1, 2, 4]:
        exog_rets = exog_data[['SPXT', 'LT11TRUU']].copy() if n_states == 2 else exog_data.copy()

        n_train = sum(exog_rets.index < TRAIN_DATE)
        train_data = exog_rets[:n_train]
        windows = range(1, len(exog_rets) - n_train + 1)

        seed = best_params[best_params.N_STATES == n_states].SEED.iloc[0] if n_states != 1 else 0
        hmm = GaussianHMM(n_states, 'full', random_state=seed, n_iter=10000, tol=1e-5, implementation='scaling')
        hmm.fit(train_data)

        A = DataFrame(hmm.transmat_)
        Q = list(hmm.predict(train_data))
        for window in windows:
            Q.append(hmm.predict(exog_rets[(n_train + window - WINDOW_SIZE):(n_train + window)])[-1])
        stock_rets['STATE'] = Q
        exog_rets['STATE'] = Q

        avg_exog_rets_q = list()
        for state in range(n_states):
            avg_exog_rets_q.append(exog_rets[exog_rets.STATE == state].mean())
        avg_exog_rets_q = concat(avg_exog_rets_q, axis=1)[:-1]
        avg_exog_rets_q.rename(index={
            'SPXT': 'Stocks',
            'DBLCIX': 'Commodities',
            'IBOXIG': 'Corporate Credit',
            'JPEICORE': 'EM Credit',
            'LT11TRUU': 'Nominal Bonds',
            'LBUTTRUU': 'IL Bonds'}, inplace=True)

        figure(figsize=(6, 3))
        plt_px = scatter(exog_px.SPXT.index[1:], exog_px.SPXT.values[1:], 0.5, Q, cmap='tab10')
        plt_w0 = axvspan(train_data.index[0], train_data.index[-1], color='grey', alpha=0.2)
        legend(labels=[f'Market Painter {state + 1}' for state in range(n_states)] + ['Training Window'],
               handles=plt_px.legend_elements()[0] + [plt_w0],
               prop={'size': 5})
        xlabel('Time')
        ylabel('Price')
        plt_spxt_px = gcf()
        close()

        figure(figsize=(6, 3))
        plt_rets = scatter(exog_rets.SPXT.index, exog_rets.SPXT.values, 0.5, Q, cmap='tab10')
        plt_w0 = axvspan(train_data.index[0], train_data.index[-1], color='grey', alpha=0.2)
        axhline(lw=0.5, c='k')
        legend(labels=[f'Market Painter {state + 1}' for state in range(n_states)] + ['Training Window'],
               handles=plt_rets.legend_elements()[0] + [plt_w0],
               prop={'size': 5},
               loc='lower left')
        xlabel('Time')
        ylabel('Returns')
        plt_spxt_rets = gcf()
        close()

        Path(FOLDER).mkdir(parents=True, exist_ok=True)
        round(A, 5).to_csv(FOLDER / f'hmm_trans_mat_q{n_states}.csv')
        round(avg_exog_rets_q, 5).to_csv(FOLDER / f'avg_exog_rets_q{n_states}.csv')
        stock_rets.to_csv(FOLDER / f'stock_rets_q{n_states}.csv')
        plt_spxt_px.savefig(FOLDER / f'plt_spxt_px_q{n_states}.pdf', bbox_inches='tight')
        plt_spxt_rets.savefig(FOLDER / f'plt_spxt_rets_q{n_states}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p1_2_hmm()