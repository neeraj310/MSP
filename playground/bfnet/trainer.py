# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from src.prefound_betas import normal_16, normal_8
from src.dataloader import normalize, generate_X
from src.bfnet import BFModel
import numpy as np
import sys

def load_betas(betas):
    return sorted([x for x in betas if x > 0])

def calculate_Y(X, betas, norm):
    Xs = X[:, 0]
    dists = []
    for x in Xs:
        dist = np.abs(np.asarray(x) - np.asarray(betas)).min()
        dists.append(dist)
    dists = normalize(dists)
    # dists = np.power((1 - dists),3)
    dists = norm(dists)
    print(dists.max())
    Y = np.array(dists)
    return Y

def train(X, Y):
    bfmodel = BFModel(6)
    bfmodel.train(X, Y)
    return bfmodel

def nn_predict(model:BFModel, X):
    return model.predict(X)

def norm(dists):
    return 1 - np.power(dists,0.5)

if __name__=="__main__":
    filename = sys.argv[1]
    beta_name = sys.argv[2]
    betas = None
    num_breaks=0
    if beta_name=="normal_8":
        betas = normal_8
        num_breaks=8
    elif beta_name =="normal_16":
        betas = normal_16
        num_breaks=16
    else:
        raise ValueError("Unsupported!")
    betas = sorted([x for x in betas if x > 0])
    X = generate_X(filename)
    Y = calculate_Y(X, betas, norm)
    bfnet = BFModel(num_breaks)
    bfnet.train(X, Y)