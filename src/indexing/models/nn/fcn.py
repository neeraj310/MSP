from timeit import default_timer as timer

import numpy as np
from sklearn import metrics

from src.indexing.learning.fully_connected_network import FullyConnectedNetwork
# from indexing.learning.pt_fcn import FullyConnectedNetwork
from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import normalize


class FCNModel(BaseModel):
    def __init__(self, page_size) -> None:
        super().__init__('Fully Connected Neural Network', page_size)
        self.net = FullyConnectedNetwork([1, 8, 1], ['relu', 'relu'], lr=0.001)

    def train(self, x_train, y_train, x_test, y_test):

        self.max_x = np.max(x_train)
        self.min_x = np.min(x_train)
        self.max_y = np.max(y_train)
        self.min_y = np.min(y_train)

        x_train = normalize(x_train)
        y_train = normalize(y_train)
        x_test = (x_test - self.min_x) / (self.max_x - self.min_x)
        y_test = (y_test - self.min_y) / (self.max_y - self.min_y)

        start_time = timer()
        self.net.fit(x_train, y_train, epochs=100, batch_size=100)
        end_time = timer()

        y_hat = self.net.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_hat)
        return mse, end_time - start_time

    def fit(self, x_train, y_train):
        self.max_x = np.max(x_train)
        self.min_x = np.min(x_train)
        self.max_y = np.max(y_train)
        self.min_y = np.min(y_train)

        x_train = normalize(x_train)
        y_train = normalize(y_train)
        self.net.fit(x_train, y_train, epochs=500, batch_size=100)

    def predict(self, X):
        X = (X - self.min_x) / (self.max_x - self.min_x)
        X = X.reshape((1))
        portion = self.net.predict(X)
        position = int(portion * (self.max_y - self.min_y)) + self.min_y
        return position // self.page_size
