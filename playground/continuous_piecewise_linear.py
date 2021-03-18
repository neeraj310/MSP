# This script shows how to train continuous piecewise linear function

import matplotlib.pyplot as plt
import numpy as np
import pwlf

from src.indexing.learning.piecewise import PiecewiseRegression
from src.indexing.utilities.metrics import mean_squared_error


def generate_x_y(number_of_pairs):
    from numpy.random import default_rng
    rng = default_rng()
    vals = rng.standard_normal(number_of_pairs)
    vals = np.abs(vals) * number_of_pairs / 10
    x = np.sort(vals)
    y = np.linspace(0, number_of_pairs - 1, number_of_pairs)
    return x, y


def pwlf_test(x, y, breaks):
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    my_pwlf.fit(breaks + 1)
    yhat = my_pwlf.predict(x)
    vis(x, y, yhat, '-', title='pwlf')
    mse = mean_squared_error(y, yhat)
    print("pwlf mean square error is {}".format(mse))


def piecewise_test(x, y, breaks):
    my_pr = PiecewiseRegression(breaks)
    my_pr.fit(x, y)
    yhat = my_pr.predict(x)
    vis(x, y, yhat, '-', title='pr')
    mse = mean_squared_error(y, yhat)
    print("mypr mean square error is {}".format(mse))


def vis(x, y, yhat, mark, title):
    plt.figure()
    plt.title(title)
    plt.plot(x, y, 'o')
    plt.plot(x, yhat, mark)
    plt.show()


if __name__ == "__main__":
    x, y = generate_x_y(10)
    # print(x)
    pwlf_test(x, y, 3)
    piecewise_test(x, y, 3)
