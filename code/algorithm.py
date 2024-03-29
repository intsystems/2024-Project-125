import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm


class Algorithm(object):
    def __init__(self, generator, total_time=400, train_window=40, a=-40, b=40):
        self.generator = generator
        self.dim = self.generator.synthesizer.dim
        self.total_time = total_time
        self.train_window = train_window
        self.a = a
        self.b = b
        self.eta = 2. / np.square(self.b - self.a)
        self.weight_const = 2.10974
        self.weights0_func = lambda x: 1 / ((x + 1) * np.square(np.log(x + 1))) / self.weight_const
        self.alpha_func = lambda x: 1 / (x + 1)
        self.loss_func = lambda x, y: np.square(x - y)
        self.signals = np.zeros((self.total_time, self.dim))
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
        # self.weights_all = np.zeros((self.total_time, self.total_time))

        # self.tramp_expert = LinearRegression()
        # self.tramp_expert_prediction = 0.
        # self.tramp_expert_loss = 0.
        # self.tramp_expert_losses_all= np.zeros(self.total_time)

    def run(self):
        for i in tqdm(range(self.total_time)):
            self.next()
            self.experts_predictions_all[i] = self.experts_predictions
            self.master_predictions_all[i] = self.master_prediction
            self.experts_losses_all[i] = self.experts_losses
            self.master_losses_all[i] = self.master_loss
            # self.weights_all[i] = self.weights

            # self.tramp_expert_losses_all[i] = self.tramp_expert_loss

    def next(self):
        if self.curr_time == 0:
            self.receive_signal()
            self.receive_response()
            self.generator.next()
            self.curr_time += 1
            return

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
        # print(f"Expert {self.curr_time+1} training on {past} - {self.curr_time}")
        self.experts.append(
            LinearRegression().fit(self.signals[past:self.curr_time], self.responses[past:self.curr_time]))
        # self.tramp_expert.fit(self.signals[past:self.curr_time], self.responses[past:self.curr_time])

    def predict(self):
        signal = self.signals[self.curr_time]
        for i, expert in enumerate(self.experts):
            self.experts_predictions[i] = expert.predict(signal.reshape(1, -1))
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
        # self.tramp_expert_loss = self.loss_func(self.tramp_expert_prediction, true_response)

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

    def draw_all(self):
        ideal_losses = np.zeros(self.total_time)
        for idx, left, right in zip(self.generator.indexes, self.generator.stamps, self.generator.stamps[1:]):
            expert_num = self.generator.stamps[idx] + self.train_window
            ideal_losses[left:right] = self.experts_losses_all.T[expert_num][left:right]

        fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True, height_ratios=[1, 2, 4])
        for stamp in self.generator.stamps:
            for i in range(3):
                ax[i].axvline(stamp, color='orange', linestyle=':')
        ax[0].plot(np.arange(self.total_time), self.responses, label="Real responses", color='black')
        ax[0].plot(np.arange(self.total_time), self.master_predictions_all, label="Master predictions", color='red')
        # ax[0].set_ylim(-15, 60)
        plt.subplots_adjust(hspace=0.2)
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Value")
        ax[0].xaxis.set_ticks_position('top')
        ax[0].legend()
        ax[1].plot(np.arange(self.total_time), self.master_losses_all, label="Master losses", color='red')
        ax[1].plot(np.arange(self.total_time), ideal_losses, label="Ideal expert losses", color='green')
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        # ax[2].plot(np.arange(self.total_time), self.tramp_expert_losses_all.cumsum(), label="Tramp expert cumulative losses",
        #            color='blue')
        ax[2].plot(np.arange(self.total_time), self.master_losses_all.cumsum(), label="Master cumulative losses",
                   color='red')
        ax[2].plot(np.arange(self.total_time), ideal_losses.cumsum(), label="Ideal expert cumulative losses",
                   color='green')
        ax[2].set_ylabel("Loss")
        ax[2].legend()

        ax2 = ax[2].twiny()
        ax2.set_xlim(ax[2].get_xlim())
        vital_indexes = [(0, self.generator.indexes[0])]
        ticks = []
        labels = []
        for i, stamp in enumerate(self.generator.stamps):
            if 0 < i < len(self.generator.indexes) and self.generator.indexes[i] != self.generator.indexes[i - 1]:
                vital_indexes.append((i, self.generator.indexes[i]))
            if i < len(self.generator.stamps) - 1:
                ticks.append((self.generator.stamps[i] + self.generator.stamps[i + 1]) / 2)
                labels.append(f"gen {self.generator.indexes[i] + 1}")
        ax2.set_xticks(ticks, labels)
        ax2.tick_params(labelcolor='orange')
        plt.show()
