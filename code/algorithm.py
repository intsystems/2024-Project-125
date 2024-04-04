import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class Algorithm(object):
    def __init__(self, series_type, generator, total_time, train_window, a, b, weights0_func=None):
        self.series_type = series_type
        self.generator = generator
        self.total_time = total_time
        self.train_window = train_window
        self.a = a
        self.b = b
        self.eta = 2. / np.square(self.b - self.a)
        self.weight_const = 2.10974
        self.weights0_func = lambda x: 1 / ((x + 1) * np.square(np.log(x + 1))) / self.weight_const \
            if weights0_func is None else weights0_func
        self.alpha_func = lambda x: 1 / (x + 1)
        self.loss_func = lambda x, y: np.square(x - y)

        self.signals = np.zeros((self.total_time, self.generator.synthesizer.dim))
        self.responses = np.zeros(self.total_time)
        self.curr_time = 0
        self.experts = []

        self.weights1 = self.weights0_func(np.arange(1, self.total_time))
        self.weights = np.r_[0, self.weights1]
        self.normalized_weights = np.zeros(self.total_time)

        self.experts_predictions = np.zeros(self.total_time)
        self.master_prediction = 0.

        self.experts_losses = np.zeros(self.total_time)
        self.master_loss = 0.

        self.experts_predictions_all = np.zeros((self.total_time, self.total_time))
        self.master_predictions_all = np.zeros(self.total_time)
        self.experts_losses_all = np.zeros((self.total_time, self.total_time))
        self.master_losses_all = np.zeros(self.total_time)

        self.segment_losses = np.full((self.generator.pieces_num, self.total_time), np.inf)
        self.best_combo = np.zeros(self.generator.pieces_num)
        self.ideal_losses = np.zeros(self.total_time)
        self.start_losses = np.zeros(self.total_time)

        self.theoretical_upper = np.zeros(self.total_time)
        self.weights_all = np.zeros((self.total_time, self.total_time))
        self.models_idxs = []

    def theoretical_upper_func(self, k, t):
        if t <= 1:
            return 0
        return (1 / self.eta *
                ((k + 1) * (2 * np.log(np.log(t)) + np.log(self.weight_const)) + (2 * k + 3) * np.log(t)))

    def run(self):
        for i in tqdm(range(self.total_time)):
            self.next()
            self.experts_predictions_all[i] = self.experts_predictions
            self.master_predictions_all[i] = self.master_prediction
            self.experts_losses_all[i] = self.experts_losses
            self.master_losses_all[i] = self.master_loss
            self.weights_all[i] = self.weights

    def next(self):
        if self.curr_time == 0:
            self.receive_signal()
            self.receive_response()
            self.generator.next()
            self.curr_time += 1

            self.experts.append("Zero expert is fiction")
            self.models_idxs.append(-1)
            return

        # if self.curr_time < 3 * self.generator.synthesizer.workers_num:
        #     self.initialize_pretrained()
        # else:
        #     self.initialize_expert()
        # self.initialize_pretrained()
        self.initialize_expert()

        self.receive_signal()
        self.normalized_weights = self.weights[1:self.curr_time + 1] / self.weights[1:self.curr_time + 1].sum()
        self.predict()

        self.receive_response()
        self.compute_losses()

        self.loss_update()
        self.mixing_update()

        self.generator.next()
        self.curr_time += 1

    def initialize_expert(self):
        past = max(self.curr_time - self.train_window, 0)
        self.experts.append(
            LinearRegression().fit(self.signals[past:self.curr_time], self.responses[past:self.curr_time]))

    def initialize_pretrained(self):
        if self.series_type == "default":
            if self.curr_time < 3 * self.generator.synthesizer.workers_num:
                idx = self.curr_time % self.generator.synthesizer.workers_num
            else:
                idx = 0
                # idx = np.random.randint(0, self.generator.synthesizer.workers_num)
            model = LinearRegression()
            model.coef_ = self.generator.params_list[idx]
            model.intercept_ = 0.
            self.experts.append(model)
            self.models_idxs.append(idx)

        if self.series_type == "arima":
            idx = np.random.randint(0, self.generator.synthesizer.workers_num)
            self.models_idxs.append(idx)
            ar, ma = self.generator.params_list[idx]
            self.experts.append((ar, ma))

    def predict(self):
        signal = self.signals[self.curr_time]
        for i, expert in enumerate(self.experts):
            if i == 0:
                continue
            if self.series_type == "default":
                self.experts_predictions[i] = expert.predict(signal.reshape(1, -1))
            if self.series_type == "arima":
                # past = max(self.curr_time - self.train_window, 0)
                # model = copy.copy(expert)
                # model.extend(self.responses[past:self.curr_time])
                # self.experts_predictions[i] = model.forecast()
                ar, ma = expert
                ar_part = self.responses[self.curr_time - 1] * ar[1] if self.curr_time > 0 else 0
                if self.curr_time < 1:
                    ma_part = 0
                else:
                    eps1 = (self.responses[self.curr_time - 1]
                            - self.experts_predictions_all[self.curr_time - 1, i])
                    ma_part = eps1 * ma[1]

                self.experts_predictions[i] = ar_part + ma_part

        self.master_prediction = self.subst() if self.experts else 0.

    def subst(self):
        e_a = np.exp(-self.eta * np.square(self.a - self.experts_predictions[1:self.curr_time + 1]))
        e_b = np.exp(-self.eta * np.square(self.b - self.experts_predictions[1:self.curr_time + 1]))
        quotient = (np.dot(self.normalized_weights, e_b)
                    / np.dot(self.normalized_weights, e_a))
        return (self.a + self.b) / 2 + np.log(quotient) / (2 * self.eta * (self.b - self.a))

    def compute_losses(self):
        true_response = self.responses[self.curr_time]
        self.experts_losses[1:self.curr_time + 1] = \
            self.loss_func(self.experts_predictions[1:self.curr_time + 1], true_response)
        self.master_loss = self.loss_func(self.master_prediction, true_response)
        self.experts_losses[self.curr_time + 1:] = self.master_loss

    def loss_update(self):
        e_l = np.exp(-self.eta * self.experts_losses[1:self.total_time])
        divisor = (np.dot(self.weights[1:self.curr_time + 1], e_l[:self.curr_time]) +
                   np.exp(-self.eta * self.master_loss) * (1 - self.weights[1:self.curr_time + 1].sum()))
        assert (1 - self.weights[1:self.curr_time + 1].sum() > 0)
        self.weights[1:self.total_time] = self.weights[1:self.total_time] * e_l / divisor

    def mixing_update(self):
        alpha = self.alpha_func(self.curr_time)
        self.weights[1:self.total_time] = (alpha * self.weights1
                                           + (1 - alpha) * self.weights[1:self.total_time])

    def receive_signal(self):
        self.signals[self.curr_time] = self.generator.get_signal()

    def receive_response(self):
        self.responses[self.curr_time] = self.generator.get_response()

    def count_segment_losses(self):
        for seg_num, left, right in zip(
                np.arange(self.generator.pieces_num),
                self.generator.stamps,
                self.generator.stamps[1:]
        ):
            for i in np.arange(1, left):
                self.segment_losses[seg_num, i] = self.experts_losses_all[left:right, i].sum()
            for i in np.arange(left, self.total_time):
                self.segment_losses[seg_num, i] = self.master_losses_all[left:right].sum()

        self.best_combo = np.argmin(self.segment_losses[:, 1:], axis=1) + 1
        for seg_num, left, right in zip(
                np.arange(self.generator.pieces_num),
                self.generator.stamps,
                self.generator.stamps[1:]
        ):
            # self.ideal_losses[left:right] = self.experts_losses_all[left:right, self.best_combo[seg_num]]
            if seg_num < self.generator.synthesizer.workers_num:
                self.ideal_losses[left:right] = self.master_losses_all[left:right]
            else:
                self.ideal_losses[left:right] = self.experts_losses_all[left:right, self.best_combo[seg_num]]

    def post_calculations(self):
        self.count_segment_losses()
        k = 0
        for idx, left, right in zip(self.generator.indexes, self.generator.stamps, self.generator.stamps[1:]):
            expert_num = self.generator.stamps[idx] + self.train_window
            self.start_losses[left:right] = self.experts_losses_all[left:right, expert_num]
            k += 1
            for t in np.arange(left, right):
                self.theoretical_upper[t] = self.theoretical_upper_func(k, t)

        self.theoretical_upper += self.ideal_losses.cumsum()

    def draw_regret(self, show=None, show_experts=None):
        if show is None:
            show = ["master"]
        if show_experts is None:
            show_experts = []
        self.post_calculations()
        plt.figure(figsize=(15, 10))
        moment = 0
        if "ideal" in show:
            plt.plot(np.arange(self.total_time)[moment:], self.ideal_losses.cumsum()[moment:],
                     label="Ideal expert cumulative losses", color='green')
        for expert_num in show_experts:
            plt.plot(np.arange(self.total_time)[moment:], self.experts_losses_all.T[expert_num].cumsum()[moment:],
                     label=f"Expert {expert_num} cumulative losses")
        if "master" in show:
            plt.plot(np.arange(self.total_time)[moment:], self.master_losses_all.cumsum()[moment:],
                     label="Master cumulative losses", color='red')
        if "theoretical" in show:
            plt.plot(np.arange(self.total_time)[moment:], self.theoretical_upper[moment:],
                     label="Theoretical upper bound", color='black')

        ax = plt.gca()
        bottom, top = ax.get_ybound()
        left, right = ax.get_xbound()
        for gen_idx, gen_stamp in zip(np.r_[self.generator.indexes, -1], self.generator.stamps):
            if gen_stamp < moment:
                continue
            ax.axvline(gen_stamp, color='orange', linestyle=':')
            if gen_idx != -1:
                ax.text(x=gen_stamp + 0.005 * (right - left), y=top - 0.05 * (top - bottom),
                        s=f"gen {gen_idx}", color='orange', rotation=15)

        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend(loc='center left')
        plt.show()

    def draw_all(self, show=None, show_experts=None):
        if show is None:
            show = ["master"]
        if show_experts is None:
            show_experts = []
        self.post_calculations()
        fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, height_ratios=[1, 2, 4])

        ax[0].plot(np.arange(self.total_time), self.responses, label="Real responses", color='black')

        if "zero" in show:
            ax[0].plot(np.arange(self.total_time), np.full(self.total_time, 0), label="Zero expert predictions",
                       color='yellow')
        for expert_num in show_experts:
            ax[0].plot(np.arange(self.total_time), self.experts_predictions_all.T[expert_num],
                       label=f"Expert {expert_num} predictions")
        if "master" in show:
            ax[0].plot(np.arange(self.total_time), self.master_predictions_all, label="Master predictions",
                       color='red')
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Value")
        ax[0].xaxis.set_ticks_position('top')
        ax[0].legend()

        plt.subplots_adjust(hspace=0.2)

        if "ideal" in show:
            ax[1].plot(np.arange(self.total_time), self.ideal_losses, label="Ideal expert losses", color='green')
        if "start" in show:
            ax[1].plot(np.arange(self.total_time), self.start_losses, label="Starter expert losses", color='purple')
        if "zero" in show:
            ax[1].plot(np.arange(self.total_time), self.loss_func(self.responses, 0), label="Zero expert losses",
                       color='yellow')
        for expert_num in show_experts:
            ax[1].plot(np.arange(self.total_time), self.experts_losses_all.T[expert_num],
                       label=f"Expert {expert_num} losses")
        if "master" in show:
            ax[1].plot(np.arange(self.total_time), self.master_losses_all, label="Master losses", color='red')

        bottom, top = ax[1].get_ybound()
        left, right = ax[1].get_xbound()
        for gen_idx, gen_stamp in zip(np.r_[self.generator.indexes, -1], self.generator.stamps):
            for i in range(3):
                ax[i].axvline(gen_stamp, color='orange', linestyle=':')
            if gen_idx != -1:
                ax[1].text(x=gen_stamp + 0.005 * (right - left), y=top - 0.12 * (top - bottom),
                           s=f"gen {gen_idx}", color='orange', rotation=15)

        ax[1].set_ylabel("Loss")
        ax[1].legend()

        if "ideal" in show:
            ax[2].plot(np.arange(self.total_time), self.ideal_losses.cumsum(),
                       label="Ideal expert cumulative losses", color='green')
        if "start" in show:
            ax[2].plot(np.arange(self.total_time), self.start_losses.cumsum(), label="Starter expert cumulative losses",
                       color='purple')
        if "zero" in show:
            ax[2].plot(np.arange(self.total_time), self.loss_func(self.responses, 0).cumsum(),
                       label="Zero expert cumulative losses", color='yellow')

        for expert_num in show_experts:
            ax[2].plot(np.arange(self.total_time), self.experts_losses_all.T[expert_num].cumsum(),
                       label=f"Expert {expert_num} cumulative losses")

        if "master" in show:
            ax[2].plot(np.arange(self.total_time), self.master_losses_all.cumsum(),
                       label="Master cumulative losses", color='red')

        if "theoretical" in show:
            ax[2].plot(np.arange(self.total_time), self.theoretical_upper,
                       label="Theoretical upper bound", color='black')

        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Regret")
        ax[2].legend()
        plt.show()
        return fig, ax
