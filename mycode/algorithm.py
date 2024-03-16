import numpy as np
# from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

class Algorithm(object):

    def __init__(self, generator, total_time=800):
        self.generator = generator
        self.total_time = total_time
        self.weights_func = lambda x: 1 / ((x + 1) * np.square(np.log(x + 1)))
        self.weight_const = 2.10974
        self.alpha_func = lambda x: 1 / (x + 1)
        self.loss_func = lambda x, y: np.square(x - y)
        # self.expert_degree = 2
        self.curr_time = 0
        # self.signals = np.zeros(self.total_time)
        self.responses = np.zeros(self.total_time)
        self.workers = self.generator.synthesizer.workers_num  # to be removed
        self.launched = 0                                      # to be removed
        self.changes = (self.generator.stamps[:self.workers]
                        + np.diff(self.generator.stamps[:self.workers+1]) // 2)
        # self.experts = np.zeros((self.total_time, self.expert_degree))
        # self.experts = np.zeros((self.workers, self.expert_degree))

        self.models = []
        # self.weights = np.zeros(self.total_time)
        # self.normalized_weights = np.zeros(self.total_time)
        self.weights = np.zeros(self.workers)
        self.normalized_weights = np.zeros(self.workers)
        self.past_window = 40
        # self.experts_predictions = np.zeros(self.total_time)
        self.experts_predictions = np.zeros(self.workers)
        self.master_prediction = 0.
        self.experts_losses = np.zeros(self.workers)
        self.master_loss = 0.
        self.a = -200.
        self.b = 200.
        self.eta = 2. / np.square(self.b - self.a)
        # self.linreg = LinearRegression(fit_intercept=False)
        self.experts_predictions_all = np.zeros((self.total_time, self.workers))
        self.master_predictions_all = np.zeros(self.total_time)
        self.experts_T =  np.zeros((self.workers, self.total_time))

    def run(self):
        for i in range(self.total_time):
            self.next()
            self.experts_predictions_all[i] = self.experts_predictions
            self.master_predictions_all[i] = self.master_prediction

    def next(self):
        # self.receive_signal()
        if self.curr_time in self.changes:
            self.initialize_expert()
        self.normalized_weights = self.weights / self.weights[:self.launched].sum()
        if self.launched > 0:
            self.predict()
            self.compute_losses()
            self.loss_update()
            self.mixing_update()
        else:
            self.receive_response()
        self.generator.next()
        self.curr_time += 1

    def initialize_expert(self):
        self.weights[self.launched] = self.weights_func(self.launched + 1)
        print(f"Expert {self.launched + 1} learning on {self.curr_time-self.past_window} - {self.curr_time}")
        self.models.append(
            ARIMA(self.responses[self.curr_time-self.past_window:self.curr_time],
                  order=(1, 0, 2), trend='ct').fit())
        self.experts_T[self.launched][self.curr_time:] \
            = self.models[-1].forecast(self.total_time - self.curr_time)
        self.launched += 1


        # k = min(self.past_window, self.curr_time-1)
        # X = self.responses[self.curr_time-k-1:self.curr_time-1].reshape(1, -1)
        # y = np.array([self.responses[self.curr_time-1]])
        # self.linreg.fit(X, y)
        # print(self.linreg.coef_, self.linreg.intercept_)
        # self.experts[self.curr_time] = np.r_[self.linreg.coef_]

    def predict(self):
        for i, model in enumerate(self.models):
            # self.experts_predictions[i] = model.predict(self.curr_time, self.curr_time)[0]
            self.experts_predictions[i] = self.experts_T[i][self.curr_time]

        self.master_prediction = self.subst()

    def subst(self):
        e_a = np.exp(-self.eta * np.square(self.a - self.experts_predictions[:self.launched]))
        e_b = np.exp(-self.eta * np.square(self.b - self.experts_predictions[:self.launched]))
        quotient = (np.dot(self.normalized_weights[:self.launched], e_b)
                    / np.dot(self.normalized_weights[:self.launched], e_a))
        # print("e_a, e_b, quotient:  ", e_a, e_b, quotient)
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




