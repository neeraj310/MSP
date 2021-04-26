# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

'''
This file requires pwlf for fast prototyping
'''
from timeit import default_timer as timer
from typing import List
import numpy as np

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import normalize
from src.indexing.models.nn.bfnet import BFModel
from src.indexing.learning.polynomial_regression import PolynomialRegression
import pwlf

class ConvModel(BaseModel):
    def __init__(self, page_size, num_breaks) -> None:
        super().__init__("Convolution b={}".format(num_breaks), page_size=page_size)
        self.num_breaks = num_breaks
        self.has_normalized = False

    def _normalize(self, x_train, y_train, x_test, y_test):
        if not self.has_normalized:
            self.max_x = np.max(x_train)
            self.min_x = np.min(x_train)
            self.max_y = np.max(y_train)
            self.min_y = np.min(y_train)
            x_train, y_train = normalize(x_train), normalize(y_train)

            if x_test is not None:
                x_test = (x_test - self.min_x) / (self.max_x - self.min_x + 1)
            if y_test is not None:
                y_test = (y_test - self.min_y) / (self.max_y - self.min_y + 1)
            return x_train, y_train, x_test, y_test
        else:
            print("has already normalized...")

    def find_locations(self, predicted_betas):
        indices = (-predicted_betas).argsort()[:self.num_breaks]
        indices = normalize(indices)
        return sorted(indices)

    def train(self, x_train, y_train, x_test, y_test):
        x_train, y_train, x_test, y_test = self._normalize(
            x_train, y_train, x_test, y_test)
        # generating X
        data = np.zeros((len(x_train),2))
        data[:, 0] = x_train[:, 0]
        data[:, 1] = y_train[:, 0]
        model = BFModel(8)
        model.load('./pretrained/bfnet.model')
        start_time = timer()
        predicted_betas = model.predict(data)
        predicted_betas = predicted_betas.reshape(y_train.shape[0],)
        indices = self.find_locations(predicted_betas)
        self.indices = indices
        self.lrs = []
        self.model = pwlf.PiecewiseLinFit(data[:, 0], data[:, 1])
        self.model.fit_with_breaks(indices)
        end_time = timer()
        yhat = []
        for each in x_test:
            yhat.append(self.predict(each))
        yhat = np.array(yhat)
        mse = metrics.mean_squared_error(y_test, yhat)
        return mse, end_time - start_time
    
    def predict(self, X):
        X = (X - self.min_x) / (self.max_x - self.min_x)
        X = np.array(X)
        output = self.model.predict(X)
        if (output<0):
            output = 0
        position = int(output * (self.max_y - self.min_y)) + self.min_y
        return position