# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pwlf

from playground.convs.fcn_net import BFModel
from src.indexing.learning.piecewise import PiecewiseRegression
from src.indexing.utilities.dataloaders import normalize

NUM_BREAKPOINTS = 6

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
    bounds = np.zeros((NUM_BREAKPOINTS, 2))
    for i in range(NUM_BREAKPOINTS):
        bounds[i, 0] = 0.0
        bounds[i, 1] = 1.0
    pwr.fit(NUM_BREAKPOINTS + 1, bounds=bounds)
    betas = [pwr.beta[line] for line in range(NUM_BREAKPOINTS + 1)]
    return betas


def fake_betas(X):
    betas = [
        -0.014831783166699978, 0.1454168912524767, 0.5895283420832912,
        0.9498482433495601, 0.8061736335468529, 0.6080531291758093,
        -0.6681222345975449, -0.6751356224716256, -0.9388678931320357
    ]
    return sorted([x for x in betas if x > 0])


def process_X(X, betas):
    Xs = X[:, 0]
    dists = []
    for x in Xs:
        dist = np.abs(np.asarray(x) - np.asarray(betas)).min()
        dists.append(dist)
    dists = normalize(dists)
    dists = np.power((1 - dists),3)
    print(dists.max())
    Y = np.array(dists)
    return Y


def visualize(X, betas, Y, indices=None):
    plt.scatter(X[:, 0], X[:, 1], linewidths=0.5)
    plt.scatter(X[:, 0], Y, marker='+')
    if not indices is None:
        plt.vlines(indices, 0, 1, colors='r', linestyles='solid')
    plt.vlines(betas, 0, 1, linestyles='dotted')
    plt.show()

def train_nn(X, Y):
    bfmodel = BFModel(6)
    bfmodel.train(X, Y)
    return bfmodel

def nn_predict(model:BFModel, X):
    return model.predict(X)

def load_nn():
    bfmodel = BFModel(6)
    bfmodel.load()
    return bfmodel


def find_locations(predicted_betas, num_breaks):
    indices = (-predicted_betas).argsort()[:num_breaks]
    indices = normalize(indices)
    return indices

if __name__ == "__main__":
    filename = sys.argv[1]
    X = generate_X(filename, None)
    betas = fake_betas(X)
    Y = process_X(X, betas)
    print(Y.shape)
    #model = train_nn(X, Y)
    model=load_nn()
    predicted_betas = nn_predict(model, X)
    predicted_betas = predicted_betas.reshape(Y.shape[0],)
    indices = find_locations(predicted_betas, NUM_BREAKPOINTS)
    print(indices)
    visualize(X, betas, Y, indices)
