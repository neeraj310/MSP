# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from src.dataloader import normalize
from src.dataloader import generate_X
import pandas as pd
import numpy as np
import pwlf
import sys

def generate_betas_pwlf(X, num_breakpoints):
    print('Generating {} betas...'.format(num_breakpoints))
    pwr = pwlf.PiecewiseLinFit(X[:, 0], X[:, 1])
    bounds = np.zeros((num_breakpoints, 2))
    for i in range(num_breakpoints):
        bounds[i, 0] = 0.0
        bounds[i, 1] = 1.0
    pwr.fit(num_breakpoints + 1, bounds=bounds)
    betas = [pwr.beta[line] for line in range(num_breakpoints + 1)]
    return betas

def process_X(X, betas):
    Xs = X[:, 0]
    dists = []
    for x in Xs:
        dist = np.abs(np.asarray(x) - np.asarray(betas)).min()
        dists.append(dist)
    dists = normalize(dists)
    dists = np.power((1 - dists), 2)
    print(dists.max())
    Y = np.array(dists)
    return Y

if __name__=="__main__":
    filename = sys.argv[1]
    num_breakpoints = int(sys.argv[2])
    outname = sys.argv[3]
    X = generate_X(filename)
    betas = generate_betas_pwlf(X, num_breakpoints)
    Y = process_X(X, betas)
    dataset = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1],'p': Y})
    dataset.to_csv('datasets/{}.csv'.format(outname))
