import random

import numpy as np
import scipy.stats as sps

class Synthesizer(object):
    def __init__(self, dim, workers_num, random_seed):
        self.rs = random_seed
        self.dim = dim
        self.workers_num = workers_num
        self.weights_list = [self.generate_random_weights(i) for i in range(self.workers_num)]

    def alternating_indexes(self, pieces_num):
        result = list(np.arange(self.workers_num))
        for i in range(pieces_num - self.workers_num):
            num = np.random.randint(self.workers_num)
            while num == result[-1]:
                num = np.random.randint(self.workers_num)
            result.append(num)
        return np.array(result)
    def generate_ts_list(self, pieces_num, lower_bound, upper_bound, alternating):
        np.random.seed(self.rs)
        if alternating:
            indexes = self.alternating_indexes(pieces_num)
        else:
            random_indexes = np.random.randint(self.workers_num, size=pieces_num - self.workers_num)
            indexes = np.r_[np.arange(self.workers_num), random_indexes]
        pieces_sizes = np.random.randint(lower_bound, upper_bound, size=pieces_num)
        ts_list = [self.generate_time_series(*self.weights_list[idx], pieces_sizes[i], self.rs+i)
                   for i, idx in enumerate(indexes)]
        return self.weights_list, indexes, ts_list

    def generate_time_series(self, weights, noise_var, n, rs):
        np.random.seed(rs)
        dim = len(weights)
        signals = np.random.normal(0, 1, size=(n, dim))
        # signals = sps.uniform(0, 1).rvs((n, dim))
        noise = np.random.normal(0, noise_var, n)
        responses = signals @ weights + noise
        return signals, responses

    def generate_random_weights(self, rs):
        np.random.seed(rs)
        noise_var = 1
        return np.random.randint(low=-20, high=20, size=self.dim), noise_var

    
