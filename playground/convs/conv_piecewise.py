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
from playground.convs.fcn_net import BFNet
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

def fake_betas(X):
    betas = [-0.014831783166699978, 0.1454168912524767, 0.5895283420832912, 0.9498482433495601, 0.8061736335468529, 0.6080531291758093, -0.6681222345975449, -0.6751356224716256, -0.9388678931320357]
    return sorted([x for x in betas if x>0])

def process_X(X, betas):
    Xs = X[:, 0]
    dists = []
    for x in Xs:
        dist = np.abs(np.asarray(x)-np.asarray(betas)).min()
        if dist == 0:
            dist = 1
        else:
            dist = 1/dist
        dists.append(dist)
    dists = normalize(dists)
    Y = np.array(dists)
    return Y

def visualize(X, betas, Y):
    plt.scatter(X[:, 0], X[:, 1], linewidths=0.5)
    plt.scatter(X[:, 0], Y, marker='+')
    plt.vlines(betas, 0, 1, linestyles='dotted')
    plt.show()

if __name__ == "__main__":
    filename = sys.argv[1]
    X = generate_X(filename, None)
    betas = fake_betas(X)
    Y = process_X(X, betas)
    print(Y)
    print(betas)
    visualize(X, betas, Y)
