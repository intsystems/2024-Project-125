import statsmodels.tsa.arima_process as arima_p
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import dataclasses


def generate_affine_time_series(weights, n):
    dim = len(weights)
    signals = np.random.normal(0, 1, size=(n, dim))
    responses = signals @ weights
    return signals, responses


def generate_arima_series(ar_params, ma_params, sigma, n):
    p = len(ar_params)
    q = len(ma_params)

    epsilon = np.random.normal(0, sigma, n + max(p, q))
    series = np.zeros(n)

    for i in range(max(p, q), n):
        ar_term = sum([ar_params[j] * series[i - j - 1] for j in range(p)])
        ma_term = sum([ma_params[j] * epsilon[i - j - 1] for j in range(q)])
        series[i] = ar_term + ma_term + epsilon[i]

    return series, epsilon


def create_ar_params():
    ar_1 = np.random.uniform(-1, 1) * 0.9
    return [ar_1]


def create_ma_params():
    ma_1 = np.random.uniform(-1, 1) * 0.9
    return [ma_1]


@dataclasses.dataclass
class GenParams:
    weights: np.array
    ar_params: list[float] | None
    ma_params: list[float] | None


class Synthesizer(object):

    def __init__(self, series_type, dim, low, high, clip,
                 noise_var, workers_num):
        self.series_type = series_type
        self.low = low
        self.high = high
        self.clip = clip
        self.noise_var = noise_var
        self.dim = dim
        self.workers_num = workers_num


    def create_indexes_and_sizes(self, length, from_start, lower_bound, upper_bound, alternating):
        pieces_indexes = np.arange(self.workers_num).tolist()
        pieces_sizes = np.random.randint(lower_bound, upper_bound, size=self.workers_num).tolist()
        curr_total = sum(pieces_sizes) if from_start else 0

        while curr_total < length:
            num = np.random.randint(self.workers_num)
            if alternating:
                while num == pieces_indexes[-1]:
                    num = np.random.randint(self.workers_num)
            pieces_indexes.append(num)

            size = np.random.randint(lower_bound, upper_bound)
            if size > length - curr_total:
                size = length - curr_total
            pieces_sizes.append(size)
            curr_total += size

        return pieces_indexes, pieces_sizes

    def generate_ts_list(self, length, from_start, lower_bound, upper_bound, alternating):
        pieces_indexes, pieces_sizes = (self.create_indexes_and_sizes
                                        (length, from_start, lower_bound, upper_bound, alternating))
        total_time = sum(pieces_sizes)
        params_list = self.generate_generators_params()
        ts_list = [self.generate_time_series(pieces_sizes[i], params_list[idx]) for i, idx in enumerate(pieces_indexes)]
        return total_time, params_list, pieces_indexes, ts_list

    def generate_generators_params(self):
        all_generators_params = []
        for i in range(self.workers_num):
            generator_params = GenParams(weights=self.generate_default_weights(),
                                         ar_params=create_ar_params() if self.series_type == "arima" else None,
                                         ma_params=create_ma_params() if self.series_type == "arima" else None)
            all_generators_params.append(generator_params)
        return all_generators_params

    def generate_default_weights(self):

        return np.random.randint(low=self.low, high=self.high, size=self.dim)

    def generate_time_series(self, n, gen_params):

        final_signals = np.empty((0, self.dim))
        final_responses = np.empty(0)
        weights = gen_params.weights
        while final_responses.size < n:
            signals, responses = generate_affine_time_series(weights, n)
            if self.series_type == 'arima':
                ar_params, ma_params = gen_params.ar_params, gen_params.ma_params
                sigma = self.noise_var
                arima_series, err = generate_arima_series(ar_params, ma_params, sigma, n)
                noise = arima_series
            else:
                noise = np.random.normal(0, self.noise_var, n)
            responses += noise
            if self.clip is not None:
                final_signals = np.r_[final_signals, signals[(self.clip[0] < responses) & (responses < self.clip[1])]]
                final_responses = np.r_[
                    final_responses, responses[(self.clip[0] < responses) & (responses < self.clip[1])]]

        return final_signals[:n], final_responses[:n]
