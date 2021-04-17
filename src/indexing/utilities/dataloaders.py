import numpy as np
import pandas as pd


def split_train_test(data, ratio=0.2):
    test_data = data.sample(n=int(ratio * len(data)))
    x_train, y_train = data.iloc[:, :-1].values, data.iloc[:, -1:].values
    x_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:,
                                                                   -1:].values
    x_index = np.linspace(0, len(x_train) - 1, len(x_train))
    x_index = x_index.reshape(-1, 1)
    return x_train, y_train, x_test, y_test, x_index

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1)

def uniform_sample(data, size=10000):
    X = data.iloc[:,0]
    Y = data.iloc[:,1]
    sampled = np.random.choice(X, size, replace=False)
    sampled = np.sort(sampled)
    ypos = np.searchsorted(X, sampled)
    Y = Y[ypos]
    sampled = pd.DataFrame({'val': sampled, 'block': Y})
    return sampled