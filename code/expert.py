from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

Expert = LinearRegression


class ArimaExpert(object):
    def __init__(self, possible=None):
        self.possible = possible
        self.linreg_model = LinearRegression()
        self.true_values = []
        self.arima_model = None
        self.ar_params = None
        self.ma_params = None

        self.restrict = lambda x: np.clip(x, -1e9, 1e9)

        self.linreg_term = 0.
        self.arima_terms = []

    def fit(self, X, y):
        self.linreg_model.fit(X, y)
        residuals = y - self.linreg_model.predict(X)
        min_loss = np.inf
        m = 0
        n = len(residuals)
        for ar_params, ma_params in self.possible:
            predicted = get_predicted(residuals, ar_params, ma_params, m, n)
            loss = np.square(residuals[m:n] - predicted).sum()
            if loss < min_loss:
                min_loss = loss
                self.ar_params = ar_params
                self.ma_params = ma_params

    def predict(self, X):
        self.linreg_term = self.linreg_model.predict(X)[0]
        arima_term = forecast_next(self.ar_params, self.ma_params, self.true_values, self.arima_terms)
        arima_term = self.restrict(arima_term)
        self.arima_terms.append(arima_term)
        return self.linreg_term + arima_term

    def update(self, response):
        self.true_values.append(response - self.linreg_term)


def forecast_next(ar_params, ma_params, true_values, predicted_values):
    ar_component = 0
    for i in range(1, len(ar_params) + 1):
        if len(predicted_values) - i >= 0:
            ar_component += ar_params[i - 1] * predicted_values[-i]

    ma_component = 0
    for i in range(1, len(ma_params) + 1):
        if len(predicted_values) - i >= 0:
            ma_component += ma_params[i - 1] * (true_values[-i] - predicted_values[-i])

    forecasted_value = ar_component - ma_component
    return forecasted_value


def get_predicted(arma_series, ar_params, ma_params, m, n):
    true_values = list(arma_series[:m])
    predicted_values = list(arma_series[:m])
    for i in range(m, n):
        forecast = forecast_next(ar_params, ma_params, true_values, predicted_values)
        predicted_values.append(forecast)
        true_values.append(arma_series[i])

    predicted_values = np.array(predicted_values)
    return predicted_values
