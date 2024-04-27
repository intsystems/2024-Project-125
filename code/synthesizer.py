import random
import statsmodels.tsa.arima_process as arima_p
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from random import uniform


def generate_linear_time_series(weights, n):
    dim = len(weights)
    signals = np.random.normal(0, 1, size=(n, dim))
    responses = signals @ weights
    return signals, responses


def generate_arma(ar_params, ma_params, sigma, n):
    p = len(ar_params)
    q = len(ma_params)

    epsilon = np.random.normal(0, sigma, n + max(p, q))
    series = np.zeros(n)

    for i in range(max(p, q), n):
        ar_term = sum([ar_params[j] * series[i - j - 1] for j in range(p)])
        ma_term = sum([ma_params[j] * epsilon[i - j - 1] for j in range(q)])
        series[i] = ar_term + ma_term + epsilon[i]

    return series, epsilon


def generate_ar_params(rs):
    """
    Generates a random AR(1) parameter within a specific range.

    Args:
        rs (int): The random seed for reproducibility.

    Returns:
        float: The generated AR(1) parameter.
    """
    random.seed(rs)
    ar_1 = uniform(-1, 1) * 0.9
    return [ar_1]


def generate_ma_params(rs):
    """
    Generates a random MA(1) parameter within a specific range.

    Args:
        rs (int): The random seed for reproducibility.

    Returns:
        float: The generated MA(1) parameter.
    """
    random.seed(rs)
    ma_1 = uniform(-1, 1) * 0.9
    return [ma_1]


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
    # ar = np.r_[1, generate_ar_params(rs + 1)]
    # ma = np.r_[1, generate_ma_params(rs + 2)]
    ar = generate_ar_params(rs + 1)
    ma = generate_ar_params(rs + 2)
    return ar, ma


class Synthesizer(object):
    """
    This class is responsible for generating synthetic time series data with different characteristics.
    """

    def __init__(self, series_type, dim, low, high, clip,
                 noise_var, workers_num, random_seed):
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
        self.arma_params = None

    def generate_indexes_and_sizes(self, length, from_start, lower_bound, upper_bound, alternating):
        """
        Generates indices and sizes for the pieces of the time series.

        Args:
            length (int): The length of the time series.
            lower_bound (int): The lower bound for the piece sizes.
            upper_bound (int): The upper bound for the piece sizes.
            alternating (bool): Whether to use alternating indices for the generators.

        Returns:
            tuple: A tuple containing the piece indices (np.array) and piece sizes (np.array).
        """
        np.random.seed(self.rs)
        pieces_indexes = np.arange(self.workers_num).tolist()
        pieces_sizes = np.random.randint(lower_bound, upper_bound, size=self.workers_num).tolist()
        curr_total = sum(pieces_sizes) if from_start else 0

        while curr_total < length:
            num = np.random.randint(self.workers_num)
            if alternating:
                while num == pieces_indexes[-1]:
                    num = np.random.randint(self.workers_num)
                pieces_indexes.append(num)
            else:
                pieces_indexes.append(num)

            size = np.random.randint(lower_bound, upper_bound)
            if size > length - curr_total:
                size = length - curr_total
            pieces_sizes.append(size)
            curr_total += size

        return pieces_indexes, pieces_sizes

    def generate_ts_list(self, length, from_start, lower_bound, upper_bound, alternating):
        """
        Generates a list of time series pieces with varying parameters.

        Args:
            length (int): The length of the time series.
            lower_bound (int): The lower bound for the piece sizes.
            upper_bound (int): The upper bound for the piece sizes.
            alternating (bool): Whether to use alternating indices for the generators.

        Returns:
            tuple: A tuple containing the parameter list, piece indices, and the list of time series pieces.
        """
        pieces_indexes, pieces_sizes = (self.generate_indexes_and_sizes
                                        (length, from_start, lower_bound, upper_bound, alternating))
        total_time = sum(pieces_sizes)

        self.params_list = [self.generate_default_weights(self.rs + i) for i in range(self.workers_num)]
        if self.series_type == "arima":
            self.arma_params = [generate_arima_params(self.rs + i) for i in range(self.workers_num)]
        ts_list = [self.generate_time_series(idx, pieces_sizes[i]) for i, idx in enumerate(pieces_indexes)]
        return total_time, self.params_list, pieces_indexes, ts_list

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

    def generate_time_series(self, idx, n):
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

        final_signals = np.empty((0, self.dim))
        final_responses = np.empty(0)
        weights = self.params_list[idx]
        while final_responses.size < n:
            signals, responses = generate_linear_time_series(weights, n)
            if self.series_type == 'arima':
                ar_params, ma_params =  self.arma_params[idx]
                sigma = self.noise_var
                arma_series, err = generate_arma(ar_params, ma_params, sigma, n)
                noise = arma_series
            else:
                noise = np.random.normal(0, self.noise_var, n)
            responses += noise
            if self.clip is not None:
                final_signals = np.r_[final_signals, signals[(self.clip[0] < responses) & (responses < self.clip[1])]]
                final_responses = np.r_[final_responses, responses[(self.clip[0] < responses) & (responses < self.clip[1])]]

        return final_signals[:n], final_responses[:n]
