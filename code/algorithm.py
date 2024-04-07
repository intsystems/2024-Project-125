import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import copy

DEFAULT_CONST = 2.10974


def default_weights_func(x):
    return 1 / ((x + 1) * np.square(np.log(x + 1))) / DEFAULT_CONST


def default_alpha_func(x):
    return 1 / (x + 1)


class Algorithm(object):
    """
    This class implements the aggregating algorithm with infinite experts (GMPP)
    for online prediction in non-stationary environments.
    """

    def __init__(self, series_type, generator, train_window,
                 total_time=None, a=None, b=None,
                 weights_func=None, weight_const=None, alpha_func=None,
                 init_pretrained=False):
        """
        Algorithm constructor

        Args:
            series_type (str): The type of time series being generated (e.g., "default", "arima").
            generator (Generator): An instance of the Generator class that provides the data stream.
            train_window (int): The size of the training window for experts.
            total_time (int, optional): The total number of time steps to run the algorithm.
                Defaults to the generator's total time.
            a (float, optional): The lower bound of the response range.
                Defaults to the minimum response value in the generator.
            b (float, optional): The upper bound of the response range.
                Defaults to the maximum response value in the generator.
            weights_func (callable, optional): A function that defines the initial weights for experts.
                Defaults to a specific decreasing function.
            alpha_func (callable, optional): A function that defines mixing update coefficients.
                Defaults to  lambda t: 1 / (t + 1)
            init_pretrained (bool, optional): Whether to initialize experts with pretrained models.
                Defaults to False.
        """

        self.series_type = series_type
        self.generator = generator
        self.train_window = train_window
        self.total_time = total_time \
            if total_time is not None else self.generator.total_time
        self.a = a if a is not None else self.generator.responses.min()
        self.b = b if b is not None else self.generator.responses.max()
        self.eta = 2. / np.square(self.b - self.a)
        self.weight_const = weight_const \
            if weight_const is not None else DEFAULT_CONST
        self.weights_func = weights_func \
            if weights_func is not None else default_weights_func
        self.alpha_func = alpha_func \
            if alpha_func is not None else default_alpha_func
        self.loss_func = lambda x, y: np.square(x - y)

        self.signals = np.zeros((self.total_time, self.generator.synthesizer.dim))
        self.responses = np.zeros(self.total_time)
        self.curr_time = 0
        self.experts = []
        self.init_pretrained = init_pretrained

        self.weights1 = self.weights_func(np.arange(1, self.total_time))
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

        self.regret = 0.

        self.theoretical_upper = np.zeros(self.total_time)
        self.weights_all = np.zeros((self.total_time, self.total_time))
        self.models_idxs = []

        self.shift = 0
        self.end = self.total_time

    def theoretical_upper_func(self, k, t):
        """
        Calculates the theoretical upper bound on the regret of the GMPP algorithm.

        Args:
            k (int): The number of experts that have been activated so far.
            t (int): The current time step.

        Returns:
            float: The theoretical upper bound on the regret.
        """
        if t <= 1:
            return 0
        return (1 / self.eta *
                ((k + 1) * (2 * np.log(np.log(t)) + np.log(self.weight_const)) + (2 * k + 3) * np.log(t)))

    def run(self):
        """
        Runs the GMPP algorithm for the specified number of time steps, saving logs
        """
        for i in tqdm(range(self.total_time)):
            self.next()
            self.experts_predictions_all[i] = self.experts_predictions
            self.master_predictions_all[i] = self.master_prediction
            self.experts_losses_all[i] = self.experts_losses
            self.master_losses_all[i] = self.master_loss
            self.weights_all[i] = self.weights

    def next(self):
        """
        Performs one iteration of the GMPP algorithm,
        including expert initialization, prediction, loss computation, and weight updates.
        """
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
        if self.init_pretrained:
            self.initialize_pretrained()
        else:
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
        """
        Initializes a new expert using linear regression on the most recent training window of data.
        """
        past = max(self.curr_time - self.train_window, 0)
        self.experts.append(
            LinearRegression().fit(self.signals[past:self.curr_time], self.responses[past:self.curr_time]))

    def initialize_pretrained(self):
        """
        Initializes a new expert using a pre-trained model from the generator, depending on the series type.
        """
        if self.series_type == "default":
            # if (9 * self.generator.synthesizer.workers_num <
            #         self.curr_time < 15 * self.generator.synthesizer.workers_num):
            #     idx = self.curr_time % self.generator.synthesizer.workers_num
            # else:
            #     idx = 0
            idx = np.random.randint(0, self.generator.synthesizer.workers_num)
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
        """
        Generates predictions from each expert and the master algorithm using the current signal and weights.
        """
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
        """
        Calculates the master prediction using the GMPP's substitution function.
        """
        e_a = np.exp(-self.eta * np.square(self.a - self.experts_predictions[1:self.curr_time + 1]))
        e_b = np.exp(-self.eta * np.square(self.b - self.experts_predictions[1:self.curr_time + 1]))
        quotient = (np.dot(self.normalized_weights, e_b)
                    / np.dot(self.normalized_weights, e_a))
        return (self.a + self.b) / 2 + np.log(quotient) / (2 * self.eta * (self.b - self.a))

    def compute_losses(self):
        """
        Computes the losses for each expert and the master algorithm based on the true response.
        """
        true_response = self.responses[self.curr_time]
        self.experts_losses[1:self.curr_time + 1] = \
            self.loss_func(self.experts_predictions[1:self.curr_time + 1], true_response)
        self.master_loss = self.loss_func(self.master_prediction, true_response)
        self.experts_losses[self.curr_time + 1:] = self.master_loss

    def loss_update(self):
        """
            Updates the weights of the experts based on their losses using the GMPP's Loss Update rule.
        """
        e_l = np.exp(-self.eta * self.experts_losses[1:self.total_time])
        divisor = (np.dot(self.weights[1:self.curr_time + 1], e_l[:self.curr_time]) +
                   np.exp(-self.eta * self.master_loss) * (1 - self.weights[1:self.curr_time + 1].sum()))

        if 1 - self.weights[1:self.curr_time + 1].sum() <= 0:
            print("self.weights[1:self.curr_time + 1].sum() = ", self.weights[1:self.curr_time + 1].sum())
        assert (1 - self.weights[1:self.curr_time + 1].sum() > 0)
        self.weights[1:self.total_time] = self.weights[1:self.total_time] * e_l / divisor

    def mixing_update(self):
        """
        Updates the weights of the experts using the AA's mixing update rule,
        which combines the initial weights with the loss-updated weights.
        """
        alpha = self.alpha_func(self.curr_time)
        self.weights[1:self.total_time] = (alpha * self.weights1
                                           + (1 - alpha) * self.weights[1:self.total_time])

    def receive_signal(self):
        """
        Receives the current signal from the generator.
        """
        self.signals[self.curr_time] = self.generator.get_signal()

    def receive_response(self):
        """
        Receives the true response from the generator.
        """
        self.responses[self.curr_time] = self.generator.get_response()

    def count_segment_losses(self):
        """
            Calculates the cumulative losses for each expert and the master algorithm
                on each segment of the data stream.
        """
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

    def count_theoretical_upper(self):
        k = 0
        for idx, left, right in zip(self.generator.indexes, self.generator.stamps, self.generator.stamps[1:]):
            # expert_num = self.generator.stamps[idx] + self.train_window
            # self.start_losses[left:right] = self.experts_losses_all[left:right, expert_num]
            if left < self.shift:
                continue
            k += 1
            for t in np.arange(left, right):
                self.theoretical_upper[t] = self.theoretical_upper_func(k, t - self.shift)

        self.theoretical_upper[self.shift:self.end] += self.ideal_losses[self.shift:self.end].cumsum()

    def post_calculations(self, from_start=True, end=None):
        """
        Performs post-processing calculations,
            including segment loss calculations and theoretical upper bound computation.
        """
        if not from_start:
            self.shift = self.generator.stamps[self.generator.synthesizer.workers_num]
        if end is not None:
            self.end = end + self.shift
        
        self.count_segment_losses()
        self.count_theoretical_upper()
        self.regret = self.master_losses_all[self.shift:self.end].sum() - self.ideal_losses[self.shift:self.end].sum()

    def draw_all(self, show=None, show_experts=None,
                 show_axes=None, height_ratios=None, suptitle=None, fig_size=(15, 10)):
        """
        Visualizes the performance of the AA algorithm,
            including predictions, losses, regrets, and theoretical bounds.

        Args:
            show (list, optional): A list of elements to display on the plots (e.g., "master", "zero", "ideal").
                Defaults to ["master"].
            show_experts (list, optional): A list of expert indices to display on the plots. Defaults to [].
            show_axes (list, optional): A list of axes to display on the plots (e.g., "value", "loss", "regret").
                Defaults to ["regret"].
            height_ratios (list, optional): A list of height ratios for the subplots.
                Defaults to [1 for _ in show_axes].
            from_start (bool, optional): Whether to display the plots
                from the beginning of the time series or after the initial experts. Defaults to True.
            fig_size (tuple, optional): Figure size. Defaults to (15,10).

        Returns:
            tuple: A tuple containing the figure and axes objects.
        """

        if show is None:
            show = ["master"]
        if show_experts is None:
            show_experts = []
        if show_axes is None:
            show_axes = ["regret"]
        if height_ratios is None:
            height_ratios = [1 for _ in show_axes]
        if suptitle is None:
            suptitle = "Algorithm"
        assert len(show_axes) == len(height_ratios), "Wrong sizes"

        text_idx = 0

        fig, axes = plt.subplots(len(show_axes), 1, figsize=fig_size, sharex=True, height_ratios=height_ratios)
        ax = [axes] if len(show_axes) == 1 else axes
        grid = np.arange(self.end - self.shift)

        try:
            idx = show_axes.index("value")

            ax[idx].plot(grid, self.responses[self.shift:self.end], label="Real responses",
                         color='black')

            if "zero" in show:
                ax[idx].plot(grid, np.full(self.end - self.shift, 0),
                             label="Zero expert predictions",
                             color='yellow')
            for expert_num in show_experts:
                ax[idx].plot(grid, self.experts_predictions_all.T[expert_num][self.shift:self.end],
                             label=f"Expert {expert_num} predictions")
            if "master" in show:
                ax[idx].plot(grid, self.master_predictions_all[self.shift:self.end],
                             label="Master predictions",
                             color='red')
            ax[idx].set_xlabel("Time")
            ax[idx].set_ylabel("Value")
            ax[idx].xaxis.set_ticks_position('top')
            ax[idx].legend()

        except ValueError:
            pass

        try:
            idx = show_axes.index("loss")
            text_idx = idx

            if "ideal" in show:
                ax[idx].plot(grid, self.ideal_losses[self.shift:self.end], label="Ideal expert losses",
                             color='green')
            if "zero" in show:
                ax[idx].plot(grid, self.loss_func(self.responses, 0)[self.shift:self.end],
                             label="Zero expert losses",
                             color='yellow')
            for expert_num in show_experts:
                ax[idx].plot(grid, self.experts_losses_all.T[expert_num][self.shift:self.end],
                             label=f"Expert {expert_num} losses")
            if "master" in show:
                ax[idx].plot(grid, self.master_losses_all[self.shift:self.end], label="Master losses",
                             color='red')

            ax[idx].set_ylabel("Loss")
            ax[idx].legend()

        except ValueError:
            pass

        try:
            idx = show_axes.index("regret")

            if "ideal" in show:
                ax[idx].plot(grid, self.ideal_losses[self.shift:self.end].cumsum(),
                             label="Ideal expert cumulative losses", color='green')
            if "zero" in show:
                ax[idx].plot(grid, self.loss_func(self.responses, 0)[self.shift:self.end].cumsum(),
                             label="Zero expert cumulative losses", color='yellow')

            for expert_num in show_experts:
                ax[idx].plot(grid, self.experts_losses_all.T[expert_num][self.shift:self.end].cumsum(),
                             label=f"Expert {expert_num} cumulative losses")

            if "master" in show:
                ax[idx].plot(grid, self.master_losses_all[self.shift:self.end].cumsum(),
                             label="Master cumulative losses", color='red')

            if "theoretical" in show:
                ax[idx].plot(grid, self.theoretical_upper[self.shift:self.end],
                             label="Theoretical upper bound", color='black')

            ax[idx].set_xlabel("Time")
            ax[idx].set_ylabel("Regret")
            ax[idx].legend()

        except ValueError:
            pass

        bottom, top = ax[text_idx].get_ybound()
        left, right = ax[text_idx].get_xbound()
        for gen_idx, gen_stamp in zip(np.r_[self.generator.indexes, -1], self.generator.stamps):
            if gen_stamp < self.shift or gen_stamp > self.end:
                continue
            for i in range(len(show_axes)):
                ax[i].axvline(gen_stamp - self.shift, color='orange', linestyle=':')
            if gen_idx != -1:
                ax[text_idx].text(x=gen_stamp - self.shift + 0.005 * (right - left), y=top - 0.12 * (top - bottom),
                                  s=f"gen {gen_idx}", color='orange', rotation=15)

        fig.suptitle(suptitle, fontsize=15)
        plt.show()
        return
