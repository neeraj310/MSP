# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
import pandas as pd

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1)

def generate_X(filename, sample=None):
    '''
    take a sample from data
    '''
    print('Generating X...')
    data = pd.read_csv(filename)
    X, Y = normalize(data.iloc[:, 0].values), normalize(data.iloc[:, 1].values)
    if sample:
        idx = np.random.choice(np.arange(len(X)), sample, replace=False)
        X = X[idx]
        Y = Y[idx]
    data = np.zeros((len(X), 2))
    data[:, 0] = X
    data[:, 1] = Y
    return data