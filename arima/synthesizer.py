import random
import statsmodels.tsa.arima_process as arima_p
import numpy as np
from random import uniform


class Synthesizer(object):
    def __init__(self, workers_num, random_seed):
        self.rs = random_seed
        self.workers_num = workers_num
        self.params_list = [self.generate_random_params(self.rs+3*i) for i in range(self.workers_num)]

    def generate_ts_list(self, pieces_num, lower_bound, upper_bound):
        np.random.seed(self.rs)
        random_indexes = np.random.randint(self.workers_num, size=pieces_num - self.workers_num)
        indexes = np.r_[np.arange(self.workers_num), random_indexes]
        pieces_sizes = np.random.randint(lower_bound, upper_bound, size=pieces_num)
        ts_list = [self.generate_time_series(*self.params_list[idx], pieces_sizes[i], self.rs+i)
                   for i, idx in enumerate(indexes)]
        return self.params_list, indexes, ts_list
    def generate_time_series(self, ar_params, ma_params, trend_coef, seasonal_coef, noise_var, n, rs):
        """
        Generate a stationary time series with trend, seasonality, and ARMA components.

        Parameters:
        ar_params (list): Coefficients of the AR model.
        ma_params (list): Coefficients of the MA model.
        trend_coef (float): Coefficient of the linear trend.
        seasonal_coef (list): Coefficients of the seasonal component.
        noise_var (float): Noise variance
        n (int): Number of data points to generate.

        Returns:
        np.array: Generated time series data.
        """
        np.random.seed(rs)
        time_series = arima_p.arma_generate_sample(ar_params, ma_params, n)
        time_series += trend_coef * np.arange(n)
        # time_series += np.power(np.arange(n), 1 + (trend_coef + 0.5) * 0.5)
        seasonal_component = np.tile(seasonal_coef, n // len(seasonal_coef) + 1)[:n]
        time_series += seasonal_component
        noise = np.random.normal(0, noise_var, n)
        time_series += noise
        return time_series

    def generate_random_params(self, rs):
        np.random.seed(rs)
        random.seed(rs)
        ar = np.r_[1, self.generate_ar_params(rs+1)]
        ma = np.r_[1, self.generate_ma_params(rs+2)]
        trend_coef = uniform(-1, 1) * 0.5
        seasonal_coef = [0] # 10 * (np.random.rand(3) - 0.5)
        noise_var = 1
        return ar, ma, trend_coef, seasonal_coef, noise_var

    def generate_ar_params(self, rs):
        random.seed(rs)
        ar_1 = uniform(-1, 1) * 0.5
        return ar_1
        # ar_2 =  uniform(-1, min(1 + ar_1, 1 - ar_1))
        # ar_params = np.r_[ar_1, ar_2]
        # ar_roots = np.roots(np.r_[1, -ar_params])
        # ar_stationary = np.all(np.abs(ar_roots) > 1)
        # return ar_params if ar_stationary else generate_ar_params()

    def generate_ma_params(self, rs):
        random.seed(rs)
        ma_1 = uniform(-1, 1) * 0.5
        # return ma_1
        ma_2 = uniform(max(-1 + ma_1, -1 - ma_1), 1) * 0.5
        ma_params = np.r_[ma_1, ma_2]
        return ma_params
        # ma_roots = np.roots(np.r_[1, ma_params])
        # ma_invertible = np.all(np.abs(ma_roots) > 1)
        # return ma_params if ma_invertible else self.generate_ma_params()

