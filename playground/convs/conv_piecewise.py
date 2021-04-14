# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.indexing.learning.piecewise import PiecewiseRegression
from src.indexing.utilities.dataloaders import normalize
import pwlf

NUM_BREAKPOINTS=16

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


def generate_betas(X):
    print('Generating betas...')
    pwr = PiecewiseRegression(NUM_BREAKPOINTS)
    pwr.fit(X[:, 0], X[:, 1])
    return pwr.betas


def generate_betas_pwlf(X):
    print('Generating {} betas...'.format(NUM_BREAKPOINTS))
    pwr = pwlf.PiecewiseLinFit(X[:, 0], X[:, 1])
    bounds = np.zeros((NUM_BREAKPOINTS,2))
    for i in range(NUM_BREAKPOINTS):
        bounds[i, 0]=0.0
        bounds[i, 1]=1.0
    pwr.fit(NUM_BREAKPOINTS+1, bounds=bounds)
    betas = [pwr.beta[line] for line in range(NUM_BREAKPOINTS+1)]
    return betas

def visualize(X, betas):
    plt.scatter(X[:, 0], X[:, 1], linewidths=0.5)
    plt.vlines(betas, 0, 1, linestyles='dotted')
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    X = generate_X(filename, None)
    betas = generate_betas_pwlf(X)
    print(betas)
    visualize(X, betas)
