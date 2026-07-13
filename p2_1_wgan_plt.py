from os import environ
from keras.utils import set_random_seed
from tensorflow import config, random
from p2_0_utils import *
from numpy import nan, mean, std, arange, c_
from keras.optimizers import Adam
from matplotlib.pyplot import ion, subplots, xlim, ylim, pause
from sklearn.linear_model import LinearRegression
from keras.backend import clear_session
from datetime import datetime
from project_config import BASE_DIR, INPUT_DIR

SEED = 42
N_STATES = 2
N_STOCKS = 8
TRAIN_DATE = '2009-06-02'
Z_DIM = 128
C_LEARN_RATE = 1e-4
G_LEARN_RATE = 1e-4
C_BETA_1 = 0.5
C_BETA_2 = 0.9
G_BETA_1 = 0.5
G_BETA_2 = 0.9
N_ITERS = 2000
LAMBDA = 10
WINDOW_SIZE = 256
BATCH_SIZE = 32
N_CRITIC = 5

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
set_random_seed(SEED)
config.experimental.enable_op_determinism()

states, stock_rets = stock_load(BASE_DIR / f'phase_1/stock_rets_q{N_STATES}.csv', N_STOCKS)
n_train = sum(stock_rets.index < TRAIN_DATE)
windows = range(len(stock_rets[n_train:]))

c_loss, g_loss = nan, nan
for state in states:  # fix hyperparameters based on the smallest data_size
    c_net = critic(N_STOCKS)
    g_net = generator(Z_DIM, N_STOCKS)

    c_opt = Adam(C_LEARN_RATE, C_BETA_1, C_BETA_2)
    g_opt = Adam(G_LEARN_RATE, G_BETA_1, G_BETA_2)

    c_losses = list()

    # PLOT >============================================================================================================

    ion()
    fig, ax = subplots()
    iterations = list()
    plt_losses = ax.scatter(iterations, c_losses, c='k', s=4)
    ax.axhline(xmax=N_ITERS, lw=0.3, c='r')
    xlim(0, N_ITERS)
    ylim(-5, LAMBDA)

    # PLOT <============================================================================================================

    for window in [0]:
        window_data = window_load(stock_rets, window, n_train, WINDOW_SIZE)
        state_data = window_data[window_data.STATE == state].drop('STATE', axis=1).values
        data_size = len(state_data)

        lin_reg = LinearRegression()
        for iteration in range(N_ITERS):
            clear_session()
            if data_size == 0:
                break

            # WGAN >====================================================================================================

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

            # WGAN <====================================================================================================

            c_loss = c_loss.numpy()
            g_loss = g_loss.numpy()
            c_losses.append(c_loss)
            c_losses_m = mean(c_losses[-100:])
            c_losses_sd = std(c_losses[-100:])

            # PLOT >====================================================================================================

            lin_reg.fit(arange(len(c_losses[-100:])).reshape(-1, 1), c_losses[-100:])
            lin_grad = abs(lin_reg.coef_[0])

            iterations.append(iteration)
            plt_losses.set_offsets(c_[iterations, c_losses])
            ax.set_title(f'Q: {state} | M: {c_losses_m:4f} | S: {c_losses_sd:4f} | G: {lin_grad:4f}')
            fig.canvas.draw_idle()
            pause(1e-2)

            # PLOT <====================================================================================================

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
