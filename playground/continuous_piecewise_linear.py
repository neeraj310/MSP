# This script shows how to train continuous piecewise linear function

import matplotlib.pyplot as plt
import numpy as np
import pwlf
from sklearn.metrics import mean_squared_error


def solve_least_sqaure(A, y):
    return np.linalg.pinv(A.T @ A) @ A.T @ y


def generate_x_y(number_of_pairs):
    from numpy.random import default_rng
    rng = default_rng()
    vals = rng.standard_normal(number_of_pairs)
    vals = np.abs(vals) * number_of_pairs / 10
    x = np.sort(vals)
    y = np.linspace(0, number_of_pairs - 1, number_of_pairs)
    return x, y


def initialize(x, breaks):
    interval = breaks
    indices = np.linspace(1, len(x) - 1, interval).astype(np.uint8)
    betas = x[indices]
    betas = np.insert(betas, 0, x[0])
    return betas


def construct_A(x, betas):
    A = [np.ones_like(x)]
    A.append(x - betas[0])
    num_of_segments = len(betas) - 1
    for i in range(num_of_segments):
        A.append(np.where(x >= betas[i + 1], x - betas[i + 1], 0))
    A = np.vstack(A).T
    return A


def calc_gradient(alpha, betas, A, y, x):
    r = A @ alpha - y
    K = np.diag(alpha)
    G = [np.ones_like(x) * -1]
    for i in range(len(betas)):
        G.append(np.where(x >= betas[i], -1, 0))
    G = np.vstack(G)
    return 2 * (K @ G @ r), 2 * (K @ G @ G.T @ K.T)


def train(x, y, batch=1, breaks=2):
    lr = 0.5
    betas = initialize(x, breaks)
    alphas = None
    for i in range(batch):
        A = construct_A(x, betas)
        alphas = solve_least_sqaure(A, y)
        first_g, second_g = calc_gradient(alphas, betas, A, y, x)
        s = -np.linalg.pinv(second_g) @ first_g
        # this needs to be proved, but there is one more element left in Y^{-1}g
        s = s[1:]
        betas = betas + lr * s
    print(alphas)
    print(betas)
    calculate_error(x, y, alphas, betas)
    return alphas, betas


def calculate_error(x, y, alphas, betas):
    A = construct_A(x, betas)
    yhat = A @ alphas
    vis(x, y, yhat, '-', title='ours')
    mse = mean_squared_error(y, yhat)
    print("mean square error is {}".format(mse))


def pwlf_test(x, y, breaks):
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    my_pwlf.fit(breaks + 1)
    yhat = my_pwlf.predict(x)
    vis(x, y, yhat, '-', title='pwlf')
    mse = mean_squared_error(y, yhat)
    print("pwlf mean square error is {}".format(mse))


def vis(x, y, yhat, mark, title):
    plt.figure()
    plt.title(title)
    plt.plot(x, y, 'o')
    plt.plot(x, yhat, mark)
    plt.show()


if __name__ == "__main__":
    x, y = generate_x_y(10)
    print(x)
    alphas, betas = train(x, y, 10000, 3)
    pwlf_test(x, y, 3)
