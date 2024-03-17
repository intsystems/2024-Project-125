import numpy as np
# from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

class Algorithm(object):

    def __init__(self, generator, total_time=200, period=40, train_window=40):
        self.generator = generator
        self.total_time = total_time
        self.period = period
        self.train_window = train_window
        self.weights_func = lambda x: 1 / ((x + 1) * np.square(np.log(x + 1)))
        self.weight_const = 2.10974
        self.alpha_func = lambda x: 1 / (x + 1)
        self.loss_func = lambda x, y: np.square(x - y)
        self.curr_time = 0
        # self.signals = np.zeros(self.total_time)
        self.responses = np.zeros(self.total_time)
        self.launched = 0                                     
        self.init_points = np.arange(self.period, self.total_time, self.period)
        self.total_experts = len(self.init_points)
        self.models = []
        self.weights = np.zeros(self.total_experts)
        self.normalized_weights = np.zeros(self.total_experts)
        self.experts_predictions = np.zeros(self.total_experts)
        self.master_prediction = 0.
        self.experts_losses = np.zeros(self.total_experts)
        self.master_loss = 0.
        self.a = -200.
        self.b = 200.
        self.eta = 2. / np.square(self.b - self.a)
        self.experts_predictions_all = np.zeros((self.total_time, self.total_experts))
        self.master_predictions_all = np.zeros(self.total_time)

    def run(self):
        for i in range(self.total_time):
            self.next()
            self.experts_predictions_all[i] = self.experts_predictions
            self.master_predictions_all[i] = self.master_prediction

    def next(self):
        # self.receive_signal()
        if self.curr_time in self.init_points:
            self.initialize_expert()
        if self.launched > 0:
            self.normalized_weights = self.weights / self.weights[:self.launched].sum()
            self.predict()
            self.compute_losses()
            self.loss_update()
            self.mixing_update()
        else:
            self.receive_response()
        # if self.curr_time % 10 == 0:
        #     for i in range(self.launched):
        #         self.models[i] = self.models[i].extend(self.responses[self.curr_time-10:self.curr_time])
        for i in range(self.launched):
            self.models[i] = self.models[i].extend(np.array([self.responses[self.curr_time]]))
        self.generator.next()
        self.curr_time += 1

    def initialize_expert(self):
        self.weights[self.launched] = self.weights_func(self.launched + 1)
        past = max(self.curr_time - self.train_window, 0)
        # print(f"Expert {self.launched + 1} learning on {past} - {self.curr_time}")
        self.models.append(
            ARIMA(self.responses[past:self.curr_time],
                  order=(1, 0, 2, )).fit())
        self.launched += 1

    def predict(self):
        for i, model in enumerate(self.models):
            self.experts_predictions[i] = model.forecast()[-1]
            # self.experts_predictions[i] = model.predict(self.curr_time, self.curr_time)[0]
        self.master_prediction = self.subst()

    def subst(self):
        e_a = np.exp(-self.eta * np.square(self.a - self.experts_predictions[:self.launched]))
        e_b = np.exp(-self.eta * np.square(self.b - self.experts_predictions[:self.launched]))
        quotient = (np.dot(self.normalized_weights[:self.launched], e_b)
                    / np.dot(self.normalized_weights[:self.launched], e_a))
        return (self.a + self.b) / 2 + np.log(quotient) / (2 * self.eta * (self.b - self.a))

    def compute_losses(self):
        true_response = self.receive_response()
        self.experts_losses = self.loss_func(self.experts_predictions[:self.launched], true_response)
        self.master_loss = self.loss_func(self.master_prediction, true_response)

    def loss_update(self):
        e_l = np.exp(-self.eta * self.experts_losses)
        divisor = (np.dot(self.weights[:self.launched], e_l) +
                   np.exp(-self.eta * self.master_loss) * (1 - self.weights[:self.launched].sum()))
        self.weights[:self.launched] = e_l / divisor

    def mixing_update(self):
        alpha = self.alpha_func(self.launched)
        self.weights[:self.launched] = (alpha * self.weights_func(1 + np.arange(self.launched))
                        + (1 - alpha) * self.weights[:self.launched])

    # def receive_signal(self):
    #     signal = self.generator.get_signal()
    #     self.signals[self.curr_time] = signal
    #     return signal

    def receive_response(self):
        response = self.generator.get_response()
        self.responses[self.curr_time] = response
        return response

