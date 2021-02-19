from timeit import default_timer as timer

from sklearn import metrics
from src.indexing.models import BaseModel
import numpy as np


class LRModel(BaseModel):
    def __init__(self) -> None:
        super().__init__("Linear Regression")
        self.model = None

    def train(self, x_train, y_train, x_test, y_test):
        start_time = timer()
        xtx = np.linalg.inv(np.matmul(x_train.T, x_train))
        hat = np.matmul(xtx, x_train.T)
        hat = np.matmul(hat, y_train)
        print(hat.shape)
        print(y_train.shape)
        self.model = np.matmul(hat, y_train)
        end_time = timer()
        predicted_y_test = np.matmul(x_test, self.model)
        mse = metrics.mean_squared_error(y_test, predicted_y_test)
        return mse, end_time - start_time

    def predict(self, key):
        return np.matmul(key, self.model)