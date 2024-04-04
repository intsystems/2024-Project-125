import random
import statsmodels.tsa.arima_process as arima_p
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from random import uniform


def generate_default_time_series(weights, clip, n, rs):
    np.random.seed(rs)
    dim = len(weights)
    signals = np.random.normal(0, 1, size=(n, dim))
    noise_var = 1
    noise = np.random.normal(0, noise_var, n)
    responses = signals @ weights + noise
    if clip is not None:
        signals = signals[(clip[0] < responses) & (responses < clip[1])]
        responses = responses[(clip[0] < responses) & (responses < clip[1])]
    return signals, responses


def generate_ar_params(rs):
    random.seed(rs)
    ar_1 = uniform(-1, 1) * 0.5
    return ar_1


def generate_ma_params(rs):
    random.seed(rs)
    ma_1 = uniform(-1, 1) * 0.5
    return ma_1


def generate_arima_params(rs):
    np.random.seed(rs)
    random.seed(rs)
    ar = np.r_[1, generate_ar_params(rs + 1)]
    ma = np.r_[1, generate_ma_params(rs + 2)]
    return ar, ma


def generate_arima_time_series(ar_params, ma_params, n, rs):
    np.random.seed(rs)
    n = n + 2
    signals = np.arange(0, n)
    # responses = arima_p.arma_generate_sample(ar_params, ma_params, n)

    epsilon = np.random.normal(0, 1, n + 1)
    responses = np.zeros(n)
    for i in range(1, n):
        ar_term = ar_params[1] * responses[i - 1]
        ma_term = ma_params[1] * epsilon[i - 1]
        responses[i] = ar_term + ma_term + epsilon[i]

    return signals[2:], responses[2:]


class Synthesizer(object):
    def __init__(self, series_type="default", dim=20,
                 low=-20, high=20, clip = None,
                 noise_var=1, workers_num=2, random_seed=18):
        self.rs = random_seed
        self.series_type = series_type
        self.low = low
        self.high = high
        self.clip = clip
        self.noise_var = noise_var
        self.dim = dim
        self.workers_num = workers_num
        self.params_list = None

    def alternating_indexes(self, pieces_num):
        np.random.seed(self.rs)
        result = list(np.arange(self.workers_num))
        for i in range(pieces_num - self.workers_num):
            num = np.random.randint(self.workers_num)
            while num == result[-1]:
                num = np.random.randint(self.workers_num)
            result.append(num)
        return np.array(result)

    def generate_indexes_and_sizes(self, pieces_num, lower_bound, upper_bound, alternating):
        np.random.seed(self.rs)
        if alternating:
            pieces_indexes = self.alternating_indexes(pieces_num)
        else:
            random_indexes = np.random.randint(self.workers_num, size=pieces_num - self.workers_num)
            pieces_indexes = np.r_[np.arange(self.workers_num), random_indexes]
        pieces_sizes = np.random.randint(lower_bound, upper_bound, size=pieces_num)
        return pieces_indexes, pieces_sizes

    def generate_ts_list(self, pieces_num, lower_bound, upper_bound, alternating):
        pieces_indexes, pieces_sizes = (self.generate_indexes_and_sizes
                                        (pieces_num, lower_bound, upper_bound, alternating))
        if self.series_type == "default":
            self.params_list = [self.generate_default_weights(self.rs + i) for i in range(self.workers_num)]
            ts_list = [generate_default_time_series(self.params_list[idx], self.clip, pieces_sizes[i], self.rs + i)
                       for i, idx in enumerate(pieces_indexes)]
            return self.params_list, pieces_indexes, ts_list
        if self.series_type == "arima":
            self.params_list = [generate_arima_params(self.rs + 3*i) for i in range(self.workers_num)]
            ts_list = [generate_arima_time_series(*self.params_list[idx], pieces_sizes[i], self.rs + i)
                       for i, idx in enumerate(pieces_indexes)]
            return self.params_list, pieces_indexes, ts_list

    def generate_default_weights(self, rs):
        np.random.seed(rs)
        return np.random.randint(low=self.low, high=self.high, size=self.dim)
