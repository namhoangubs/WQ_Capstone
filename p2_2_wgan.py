from os import environ
from keras.utils import set_random_seed
from tensorflow import config, random
from p2_0_utils import *
from pathlib import Path
from numpy import nan, mean, std, arange
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from keras.backend import clear_session
from datetime import datetime
from pandas import DataFrame
from matplotlib.pyplot import figure, plot, legend, axhline, xlabel, ylabel, axvspan, gcf, close
from pathlib import Path
from project_config import BASE_DIR, INPUT_DIR

def p2_2_wgan(n_states=1, n_stocks=8):
    SEED = 42
    TRAIN_DATE = '2009-06-02'
    Z_DIM = 128
    C_LEARN_RATE = 1e-4
    G_LEARN_RATE = 1e-4
    C_BETA_1 = 0.5
    C_BETA_2 = 0.9
    G_BETA_1 = 0.5
    G_BETA_2 = 0.9
    WINDOW_SIZE = 256
    N_ITERS = 2000
    BATCH_SIZE = 32
    N_CRITIC = 5
    LAMBDA = 10
    FOLDER = BASE_DIR / 'phase_2'

    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_random_seed(SEED)
    config.experimental.enable_op_determinism()

    states, stock_rets = stock_load(BASE_DIR / f'phase_1/stock_rets_q{n_states}.csv', n_stocks)
    n_train = sum(stock_rets.index < TRAIN_DATE)
    windows = range(len(stock_rets[n_train:]))

    Path(BASE_DIR / 'models').mkdir(parents=True, exist_ok=True)
    report, c_loss, g_loss = list(), nan, nan
    for state in states:
        c_net = critic(n_stocks)
        g_net = generator(Z_DIM, n_stocks)

        c_opt = Adam(C_LEARN_RATE, C_BETA_1, C_BETA_2)
        g_opt = Adam(G_LEARN_RATE, G_BETA_1, G_BETA_2)

        c_losses, c_losses_m, c_losses_sd, lin_grad = list(), nan, nan, nan
        for window in windows:
            window_data = window_load(stock_rets, window, n_train, WINDOW_SIZE)
            state_data = window_data[window_data.STATE == state].drop('STATE', axis=1).values
            data_size = len(state_data)

            lin_reg = LinearRegression()
            for iteration in range(N_ITERS):
                clear_session()
                if data_size == 0:
                    report.append({
                        'STATE': state,
                        'WINDOW': window,
                        'LENGTH': data_size,
                        'ITERATION': iteration,
                        'C_LOSS': c_loss,
                        'G_LOSS': g_loss})

                    print(f'Q: {state} of {states[-1]} |',
                          f'W: {window:4d} of {windows[-1]} |',
                          f'N: {data_size:4d} |',
                          f'I: {iteration:4d} of {N_ITERS - 1} |',
                          f'C: {c_loss:{" " if c_loss > 0 else ""}.4f} |',
                          f'M: {c_losses_m:{" " if c_losses_m > 0 else ""}.4f} |',
                          f'S: {c_losses_sd:4f} |',
                          f'G: {lin_grad:4f} |',
                          datetime.now().replace(microsecond=0))
                    break

                # WGAN >================================================================================================

                real_data = state_data[choice(data_size, BATCH_SIZE)]
                for _ in range(N_CRITIC):
                    z = random.normal((BATCH_SIZE, Z_DIM))
                    fake_data = g_net(z)
                    with GradientTape() as tape:
                        real_score = c_net(real_data)
                        fake_score = c_net(fake_data)
                        c_cost = reduce_mean(fake_score) - reduce_mean(real_score)
                        c_loss = c_cost + LAMBDA * grad_penalty(real_data, fake_data, c_net)

                    c_grad = tape.gradient(c_loss, c_net.trainable_variables)
                    c_opt.apply_gradients(zip(c_grad, c_net.trainable_variables))

                z = random.normal((BATCH_SIZE, Z_DIM))
                with GradientTape() as tape:
                    fake_data = g_net(z)
                    fake_score = c_net(fake_data)
                    g_loss = -reduce_mean(fake_score)

                g_grad = tape.gradient(g_loss, g_net.trainable_variables)
                g_opt.apply_gradients(zip(g_grad, g_net.trainable_variables))

                # WGAN <================================================================================================

                c_loss = c_loss.numpy()
                g_loss = g_loss.numpy()
                c_losses.append(c_loss)
                c_losses_m = mean(c_losses[-100:])
                c_losses_sd = std(c_losses[-100:])

                lin_reg.fit(arange(len(c_losses[-100:])).reshape(-1, 1), c_losses[-100:])
                lin_grad = abs(lin_reg.coef_[0])

                report.append({
                    'STATE': state,
                    'WINDOW': window,
                    'LENGTH': data_size,
                    'ITERATION': iteration,
                    'C_LOSS': c_loss,
                    'G_LOSS': g_loss})

                print(f'Q: {state} of {states[-1]} |',
                      f'W: {window:4d} of {windows[-1]} |',
                      f'N: {data_size:4d} |',
                      f'I: {iteration:4d} of {N_ITERS - 1} |',
                      f'C: {c_loss:{" " if c_loss > 0 else ""}.4f} |',
                      f'M: {c_losses_m:{" " if c_losses_m > 0 else ""}.4f} |',
                      f'S: {c_losses_sd:4f} |',
                      f'G: {lin_grad:4f} |',
                      datetime.now().replace(microsecond=0))

                if break_check(window, iteration, c_losses_m, c_losses_sd, lin_grad):
                    break

            g_net.save_weights(BASE_DIR / 'models' / f'w{window}_s{state}_q{n_states}.weights.h5')

    # PLOT >============================================================================================================

    loss_report = DataFrame(report)

    plt_loss_w0, plt_loss = list(), list()
    for state in states:
        loss_q = loss_report[loss_report.STATE == state]

        loss_q_w0 = loss_q[loss_q.WINDOW == 0]
        c_loss_q_w0 = loss_q_w0.C_LOSS.to_list()
        g_loss_q_w0 = loss_q_w0.G_LOSS.to_list()

        loss_q_w = loss_q[loss_q.WINDOW != 0]
        c_loss_q_w = [loss_q[loss_q.WINDOW == window].iloc[-1].C_LOSS for window in unique(loss_q_w.WINDOW)]
        g_loss_q_w = [loss_q[loss_q.WINDOW == window].iloc[-1].G_LOSS for window in unique(loss_q_w.WINDOW)]

        figure(figsize=(4, 3))
        plot(c_loss_q_w0, lw=0.3, c='c')
        plot(g_loss_q_w0, lw=0.3, c='g')
        legend(labels=['Critic Loss', 'Generator Loss'])
        axhline(lw=0.5, c='k')
        xlabel('Iteration')
        ylabel('Loss')
        plt_loss_w0.append(gcf())
        close()

        figure(figsize=(4, 3))
        plot(c_loss_q_w0 + c_loss_q_w, lw=0.3, c='c')
        plot(g_loss_q_w0 + g_loss_q_w, lw=0.3, c='g')
        axvspan(0, len(c_loss_q_w0), color='grey', alpha=0.2)
        legend(labels=['Critic Loss', 'Generator Loss', 'Training Window'])
        axhline(lw=0.5, c='k')
        xlabel('Iteration')
        ylabel('Loss')
        plt_loss.append(gcf())
        close()

    # PLOT <============================================================================================================

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    loss_report.to_csv(FOLDER / f'loss_report_q{n_states}_n{n_stocks}.csv', index=False)
    stock_rets.to_csv(FOLDER / f'stock_rets_q{n_states}_n{n_stocks}.csv')
    for state in states:
        plt_loss_w0[state].savefig(FOLDER / f'plt_loss_w0_s{state}_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')
        plt_loss[state].savefig(FOLDER / f'plt_loss_s{state}_q{n_states}_n{n_stocks}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    p2_2_wgan()
