from timeit import default_timer as timer
import sys
from sklearn import metrics

from src.indexing.learning.polynomial_regression import PolynomialRegression
from src.indexing.models import BaseModel


class PRModel(BaseModel):
    def __init__(self, degree) -> None:
        super().__init__("Polynomial Regression with degree {}".format(degree))
        self.model = PolynomialRegression(degree)

    def train(self, x_train, y_train, x_test, y_test):
        start_time = timer()
        self.model.fit(x_train, y_train)
        end_time = timer()
        yhat = self.model.predict(x_test)
        mse = metrics.mean_squared_error(y_test, yhat)
        return mse, end_time - start_time

    def predict(self, key):
        return self.model.predict(key)