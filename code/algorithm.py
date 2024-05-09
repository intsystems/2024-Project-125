import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from experts import Expert, ArimaExpert

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
                 weights_func=None, weight_const=None,
                 mixing_type="start", alpha_func=None):
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
        self.mixing_type = mixing_type
        self.loss_func = lambda x, y: np.square(x - y)

        self.signals = np.zeros((self.total_time, self.generator.synthesizer.dim))
        self.responses = np.zeros(self.total_time)
        self.curr_time = 0
        self.experts = []

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

        self.weights_all = np.zeros((self.total_time, self.total_time))

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
            return

        self.initialize_expert()

        self.receive_signal()
        self.normalized_weights = self.weights[1:self.curr_time + 1] / self.weights[1:self.curr_time + 1].sum()
        self.predict()

        self.receive_response()
        self.compute_losses()
        if self.series_type == "arima":
            for i, expert in enumerate(self.experts):
                if i == 0:
                    continue
                expert.update(response=self.responses[self.curr_time])

        self.loss_update()
        self.mixing_update()

        self.generator.next()
        self.curr_time += 1

    def initialize_expert(self):
        """
        Initializes a new expert using linear regression on the most recent training window of data.
        """
        past = int(max(self.curr_time - self.train_window, 0))

        if self.series_type == "arima":
            expert = ArimaExpert()
        else:
            expert = Expert()

        expert.fit(self.signals[past:self.curr_time], self.responses[past:self.curr_time])
        self.experts.append(expert)

    def predict(self):
        """
        Generates predictions from each expert and the master algorithm using the current signal and weights.
        """
        signal = self.signals[self.curr_time]
        for i, expert in enumerate(self.experts):
            if i == 0:
                continue
            self.experts_predictions[i] = expert.predict(signal.reshape(1, -1))

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
        # self.experts_losses[:10] = 1e6
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
        if self.mixing_type == "start":
            self.weights[1:self.total_time] = (alpha * self.weights1
                                               + (1 - alpha) * self.weights[1:self.total_time])
        if self.mixing_type == "uniform past":
            uniform_component = self.weights_all[:self.curr_time, 1:self.total_time].mean(axis=0)
            self.weights[1:self.total_time] = (alpha * uniform_component
                                               + (1 - alpha) * self.weights[1:self.total_time])
        if self.mixing_type == "decaying past":
            gamma = 1
            mixing = (self.curr_time - np.arange(self.curr_time)) ** gamma
            mixing_weights = mixing / mixing.sum()
            decaying_component = (mixing_weights * self.weights_all[:self.curr_time, 1:self.total_time].T).sum(axis=1)
            self.weights[1:self.total_time] = (alpha * decaying_component
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
