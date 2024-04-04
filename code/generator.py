import numpy as np
import matplotlib.pyplot as plt


def merge_default_time_series(ts_list):
    merged_signals = np.concatenate([s for s, r in ts_list], axis=0)
    merged_responses = np.concatenate([r for s, r in ts_list])
    stamps = np.r_[0, np.cumsum([r.size for s, r in ts_list])]
    return merged_signals, merged_responses, stamps


def get_edges_means(time_series: np.array, indent=4):
    left = time_series[0:indent].mean()
    right = time_series[-indent:].mean()
    return left, right


def merge_arima_time_series(ts_list: list[np.array]):
    merged_signals = np.concatenate([s for s, r in ts_list], axis=0)
    merged_responses = np.concatenate([r for s, r in ts_list])
    stamps = np.r_[0, np.cumsum([r.size for s, r in ts_list])]
    return merged_signals, merged_responses, stamps
    # total_len = sum(len(r) for s, r in ts_list)
    # merged_signals = np.concatenate([s for s, r in ts_list], axis=0)
    # merged_responses = np.zeros(total_len)
    # start_idx = 0
    # last_right = 0
    # shifts = []
    # stamps = [0]
    # for s, resps in ts_list:
    #     end_idx = start_idx + len(resps)
    #     left, right = get_edges_means(resps)
    #     shift = last_right - left
    #     merged_responses[start_idx:end_idx] = resps + shift
    #     last_right = right + shift
    #     shifts.append(shift)
    #     stamps.append(end_idx)
    #     start_idx = end_idx
    # return merged_signals, merged_responses, np.array(stamps)


class Generator(object):
    def __init__(self, series_type, synthesizer):
        self.series_type = series_type
        self.synthesizer = synthesizer
        self.curr_time = 0
        self.total_time = 0
        self.signals = None
        self.responses = None
        self.params_list = None
        self.ts_list = None
        self.pieces_num = None
        self.indexes = None
        self.stamps = None

    def generate(self, pieces_num=10, lower_bound=80, upper_bound=100, alternating=True):
        self.pieces_num = pieces_num
        self.params_list, self.indexes, self.ts_list = \
            self.synthesizer.generate_ts_list(pieces_num, lower_bound, upper_bound, alternating)
        if self.series_type == "default":
            self.signals, self.responses, self.stamps = merge_default_time_series(self.ts_list)
        if self.series_type == "arima":
            self.signals, self.responses, self.stamps = merge_arima_time_series(self.ts_list)
        self.total_time = len(self.responses)

    def launch(self):
        self.curr_time = 0

    def next(self):
        self.curr_time += 1

    def get_signal(self):
        return self.signals[self.curr_time]

    def get_response(self):
        return self.responses[self.curr_time]

    def show_time_series(self):
        k = self.synthesizer.workers_num
        fig, ax = plt.subplots(k, 1, figsize=(15, 15))
        for i in range(k):
            s, ts = self.ts_list[i]
            ax[i].plot(np.arange(len(ts)), ts, color='black')
            ax[i].set_title(f"Piece made by generator №{i + 1}")
        plt.show()

    def draw_merged(self):
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax1.plot(np.arange(self.total_time), self.responses, color='black')
        ax2 = ax1.twiny()
        vital_indexes = [(0, self.indexes[0])]
        ticks = []
        labels = []
        for i, stamp in enumerate(self.stamps):
            if 0 < i < len(self.indexes) and self.indexes[i] != self.indexes[i - 1]:
                vital_indexes.append((i, self.indexes[i]))
            if i < len(self.stamps) - 1:
                ticks.append((self.stamps[i] + self.stamps[i+1]) / 2)
                labels.append(f"generator {self.indexes[i]+1}")
        for i, idx in vital_indexes:
            plt.axvline(self.stamps[i], color='orange')
        ax2.axvline(self.stamps[-1], color='orange')
        ax2.set_xticks(ticks, labels)
        ax2.tick_params(labelcolor='orange')
        plt.title("Merged time series")
        plt.show()
