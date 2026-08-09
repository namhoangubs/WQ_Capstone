from pandas import read_csv, merge
from numpy.random import choice
from numpy import unique, abs, float64, mean, std
from keras import Sequential
from keras.layers import Input, Dense, Dropout
from tensorflow import random, GradientTape, reduce_mean, square, norm


import tensorflow as tf
from project_config import BASE_DIR, INPUT_DIR

class Maxout(tf.keras.layers.Layer):
    """Parameter-free replacement for tensorflow_addons.layers.Maxout.

    TensorFlow Addons Maxout groups the existing final-axis features and takes
    the maximum within each group.  It does not add a Dense projection.  Keeping
    that behavior is important here because the critic already contains the
    4096-feature Dense layer used by the original implementation.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)

    def call(self, inputs):
        feature_count = inputs.shape[-1]
        if feature_count is None:
            raise ValueError("Maxout requires a statically known feature dimension.")
        if int(feature_count) % self.units:
            raise ValueError(
                f"Input feature count {feature_count} is not divisible by "
                f"the requested {self.units} maxout units."
            )
        pieces = int(feature_count) // self.units
        output_shape = tf.concat(
            [
                tf.shape(inputs)[:-1],
                tf.constant([self.units, pieces], dtype=tf.int32),
            ],
            axis=0,
        )
        grouped = tf.reshape(inputs, output_shape)
        return tf.reduce_max(grouped, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

def stock_load(data_path, n_stocks):
    s_data = read_csv(data_path, index_col=[0], parse_dates=[0])
    Q = s_data.pop('STATE')
    s_tickers = choice(s_data.columns, n_stocks, False)
    return range(len(unique(Q))), merge(s_data[s_tickers], Q, on='DATE')


def window_load(data, window, n_train, window_size):
    if window == 0:
        w_data = data[:n_train]
    else:
        w_data = data[(n_train + window - window_size):(n_train + window)]
    return w_data


def critic(n_stocks):
    c_net = Sequential()
    c_net.add(Input((n_stocks,)))

    c_net.add(Dense(4096, 'tanh'))
    c_net.add(Dropout(0.25))
    c_net.add(Maxout(256))

    c_net.add(Dense(64, 'tanh'))
    c_net.add(Dropout(0.25))
    c_net.add(Maxout(4))

    c_net.add(Dense(1))
    return c_net


def generator(z_dim, n_stocks, output_activation='tanh'):
    g_net = Sequential()
    g_net.add(Input((z_dim,)))

    g_net.add(Dense(256, 'tanh'))
    g_net.add(Dense(512, 'tanh'))
    g_net.add(Dense(1024, 'tanh'))
    g_net.add(Dense(2048, 'tanh'))
    g_net.add(Dense(4096, 'tanh'))

    g_net.add(Dense(n_stocks, output_activation))
    return g_net


def grad_penalty(real_data, fake_data, c_model):
    interpolation_shape = tf.stack([
        tf.shape(real_data)[0],
        tf.constant(1, dtype=tf.int32),
    ])
    eps = random.uniform(interpolation_shape, dtype=real_data.dtype)
    est = eps * real_data + (1 - eps) * fake_data

    with GradientTape() as tape:
        tape.watch(est)
        # Match the critic mode used for the real and generated scores.
        c_est = c_model(est, training=False)

    grad = tape.gradient(c_est, est)
    return reduce_mean(square(norm(grad, axis=1) - 1))


def break_check(window, iteration, c_losses_m, c_losses_sd, lin_grad):
    stop = False
    if window == 0 and iteration > 0 and lin_grad < 1e-5 and abs(c_losses_m) < 1e-1:
        stop = True
    if window == 0 and iteration > 0 and lin_grad < 1e-4 and abs(c_losses_m) < 1e-1 and c_losses_sd < 5e-2:
        stop = True
    if window == 0 and iteration > 0 and lin_grad < 1e-4 and abs(c_losses_m) < 1e-2:
        stop = True
    if window == 0 and iteration > 999 and lin_grad < 1e-4 and abs(c_losses_m) < 2e-1:
        stop = True
    if window > 0 and iteration > 0 and lin_grad < 1e-5:
        stop = True
    return stop


def course_fine_vol(data):
    c_vol, f_vol = float64(), float64()
    for day in range(1, 6):
        c_vol += data.shift(day)
        f_vol += abs(data.shift(day))
    return abs(c_vol).dropna(), f_vol.dropna()


def course_fine_corr(data, lag):
    c_vol, f_vol = course_fine_vol(data)
    num = mean((c_vol.shift(-lag) - mean(c_vol)) * (f_vol - mean(f_vol)))
    den = std(c_vol) * std(f_vol)
    return num / den
