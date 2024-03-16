import numpy as np
import matplotlib.pyplot as plt

class Generator(object):
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.curr_time = 0
        self.total_time = 0
        self.responses = None
        self.params_list = None
        self.ts_list = None
        self.indexes = None
        self.stamps = None

    def generate(self, pieces_num, lower_bound, upper_bound):
        self.params_list, self.indexes, self.ts_list = \
            self.synthesizer.generate_ts_list(pieces_num, lower_bound, upper_bound)
        self.responses, self.stamps = self.bind_time_series(self.ts_list)
        self.total_time = len(self.responses)

    def next(self):
        if self.curr_time + 1 < self.total_time:
            self.curr_time += 1
        else:
            raise IndexError("Time is up!")

    def get_signal(self):
        return self.curr_time

    def get_response(self):
        return self.responses[self.curr_time]

    def bind_time_series(self, time_series_list: list[np.array]):
        total_len = sum(len(ts) for ts in time_series_list)
        merged_time_series = np.zeros(total_len)
        start_idx = 0
        last_right = 0
        shifts = []
        stamps = [0]
        for ts in time_series_list:
            end_idx = start_idx + len(ts)
            left, right = self.get_edges_means(ts)
            shift = last_right - left
            merged_time_series[start_idx:end_idx] = ts + shift
            last_right = right + shift
            shifts.append(shift)
            stamps.append(end_idx)
            start_idx = end_idx
        return merged_time_series, np.array(stamps)

    def get_edges_means(self, time_series: np.array, indent=4):
        left = time_series[0:indent].mean()
        right = time_series[-indent:].mean()
        return left, right

    def show_time_series(self):
        k = self.synthesizer.workers_num
        fig, ax = plt.subplots(k, 1, figsize=(15, 15))
        for i in range(k):
            ts = self.ts_list[i]
            ax[i].plot(np.arange(len(ts)), ts)
        plt.show()

    def draw_binded(self):
        plt.figure(figsize=(15, 6))
        plt.plot(np.arange(self.total_time), self.responses)
        vital_indexes = [(0, self.indexes[0])]
        for i, stamp in enumerate(self.stamps):
            if 0 < i < len(self.indexes) and self.indexes[i] != self.indexes[i - 1]:
                vital_indexes.append((i, self.indexes[i]))
        ticks = []
        labels = []

        for i, idx in vital_indexes:
            plt.axvline(self.stamps[i], color='red')
            ticks.append(self.stamps[i])
            labels.append(f"     {idx} =>")
        plt.axvline(self.stamps[-1], color='red')
        plt.xticks(ticks, labels)
        plt.show()

