import random
import statsmodels.tsa.arima_process as arima_p
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from random import uniform


def generate_default_time_series(weights, clip, n, rs):
    """
    Generates a synthetic time series using a linear model with Gaussian noise.

    Args:
        weights (np.array): The weight vector for the linear model.
        clip (tuple, optional): A tuple (min, max) specifying the range to clip the responses. Defaults to None.
        n (int): The length of the time series.
        rs (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the generated signals (np.array) and responses (np.array).
    """
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
    """
    Generates a random AR(1) parameter within a specific range.

    Args:
        rs (int): The random seed for reproducibility.

    Returns:
        float: The generated AR(1) parameter.
    """
    random.seed(rs)
    ar_1 = uniform(-1, 1) * 0.5
    return ar_1


def generate_ma_params(rs):
    """
    Generates a random MA(1) parameter within a specific range.

    Args:
        rs (int): The random seed for reproducibility.

    Returns:
        float: The generated MA(1) parameter.
    """
    random.seed(rs)
    ma_1 = uniform(-1, 1) * 0.5
    return ma_1


def generate_arima_params(rs):
    """
    Generates random ARIMA(1, 0, 1) parameters.

    Args:
        rs (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the AR parameters (np.array) and MA parameters (np.array).
    """
    np.random.seed(rs)
    random.seed(rs)
    ar = np.r_[1, generate_ar_params(rs + 1)]
    ma = np.r_[1, generate_ma_params(rs + 2)]
    return ar, ma


def generate_arima_time_series(ar_params, ma_params, n, rs):
    """
    Generates a synthetic time series using an ARIMA(1, 0, 1) model.

    Args:
        ar_params (np.array): The AR parameters of the ARIMA model.
        ma_params (np.array): The MA parameters of the ARIMA model.
        n (int): The length of the time series.
        rs (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the generated signals (np.array) and responses (np.array).
    """
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
    """
    This class is responsible for generating synthetic time series data with different characteristics.
    """
    def __init__(self, series_type, dim, low, high, clip = None,
                 noise_var=1, workers_num=2, random_seed=18):
        """
        Synthesizer constructor

        Args:
            series_type (str, optional): The type of time series to generate (e.g., "default", "arima").
            dim (int, optional): The dimensionality of the signals.
            low (int, optional): The lower bound for generating random weights.
            high (int, optional): The upper bound for generating random weights.
            clip (tuple, optional): A tuple (min, max) specifying the range to clip the responses. Defaults to None.
            noise_var (int, optional): The variance of the Gaussian noise added to the responses. Defaults to 1.
            workers_num (int, optional): The number of different generators to simulate. Defaults to 2.
            random_seed (int, optional): The random seed for reproducibility. Defaults to 18.
        """

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
        """
        Generates a sequence of alternating indices for the generators.

        Args:
            pieces_num (int): The number of pieces in the time series.

        Returns:
            np.array: An array of alternating indices.
        """
        np.random.seed(self.rs)
        result = list(np.arange(self.workers_num))
        for i in range(pieces_num - self.workers_num):
            num = np.random.randint(self.workers_num)
            while num == result[-1]:
                num = np.random.randint(self.workers_num)
            result.append(num)
        return np.array(result)

    def generate_indexes_and_sizes(self, pieces_num, lower_bound, upper_bound, alternating):
        """
        Generates indices and sizes for the pieces of the time series.

        Args:
            pieces_num (int): The number of pieces in the time series.
            lower_bound (int): The lower bound for the piece sizes.
            upper_bound (int): The upper bound for the piece sizes.
            alternating (bool): Whether to use alternating indices for the generators.

        Returns:
            tuple: A tuple containing the piece indices (np.array) and piece sizes (np.array).
        """
        np.random.seed(self.rs)
        if alternating:
            pieces_indexes = self.alternating_indexes(pieces_num)
        else:
            random_indexes = np.random.randint(self.workers_num, size=pieces_num - self.workers_num)
            pieces_indexes = np.r_[np.arange(self.workers_num), random_indexes]
        pieces_sizes = np.random.randint(lower_bound, upper_bound, size=pieces_num)
        return pieces_indexes, pieces_sizes

    def generate_ts_list(self, pieces_num, lower_bound, upper_bound, alternating):
        """
        Generates a list of time series pieces with varying parameters.

        Args:
            pieces_num (int): The number of pieces in the time series.
            lower_bound (int): The lower bound for the piece sizes.
            upper_bound (int): The upper bound for the piece sizes.
            alternating (bool): Whether to use alternating indices for the generators.

        Returns:
            tuple: A tuple containing the parameter list, piece indices, and the list of time series pieces.
        """
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
        """
        Generates random weights for the linear model.

        Args:
            rs (int): The random seed for reproducibility.

        Returns:
            np.array: An array of random weights.
        """
        np.random.seed(rs)
        return np.random.randint(low=self.low, high=self.high, size=self.dim)
