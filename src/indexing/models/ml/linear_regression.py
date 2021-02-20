from timeit import default_timer as timer

from sklearn import metrics
from src.indexing.models import BaseModel
from src.indexing.learning.linear_regression import LinearRegression
# from sklearn.linear_model import LinearRegression
import numpy as np

class LRModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("Linear Regression")
        self.model = None

    def train(self, x_train, y_train, x_test, y_test):
        start_time = timer()
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        end_time = timer()
        self.model = lr
        pred_func = np.vectorize(self.model.predict)
        yhat = pred_func(x_test)
        mse = metrics.mean_squared_error(y_test, yhat)
        return mse, end_time - start_time

    def predict(self, key):
        return self.model.predict(key)