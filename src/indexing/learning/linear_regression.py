import numpy as np


class LinearRegression():
    def __init__(self) -> None:
        self.coeffs = None

    def __repr__(self) -> str:
        return "[w={0:.2},w_0={0:.2}]".format(float(self.w), float(self.b))

    def fit(self, x, y):
        numerator = np.mean(np.multiply(x, y)) - np.mean(x) * np.mean(y)
        denominator = np.mean(np.multiply(x, x)) - np.mean(x) * np.mean(x)
        self.w = numerator / denominator
        self.b = np.mean(y) - np.mean(x) * self.w

    def predict(self, x):
        return self.w * x + self.b
