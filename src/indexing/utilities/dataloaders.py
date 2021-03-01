import numpy as np


def split_train_test(data, ratio=0.2):
    test_data = data.sample(n=int(ratio * len(data)))
    x_train, y_train = data.iloc[:, :-1].values, data.iloc[:, -1:].values
    x_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1:].values
    return x_train, y_train, x_test, y_test


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
