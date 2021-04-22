# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from src.dataloader import normalize, generate_X
from src.bfnet import BFModel
from src.prefound_betas import normal_16, normal_8
from src.trainer import calculate_Y, norm
import matplotlib.pyplot as plt
import sys

def visualize(X, betas, Y, indices=None):
    plt.scatter(X[:, 0], X[:, 1], linewidths=0.5)
    plt.scatter(X[:, 0], Y, marker='+')
    if not indices is None:
        plt.vlines(indices, 0, 1, colors='r', linestyles='solid')
    plt.vlines(betas, 0, 1, linestyles='dotted')
    plt.show()

def find_locations(predicted_betas, num_breaks):
    indices = (-predicted_betas).argsort()[:num_breaks]
    indices = normalize(indices)
    return indices

def load():
    bfmodel = BFModel(6)
    bfmodel.load()
    return bfmodel


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
    model=load()
    predicted_betas = model.predict(X)
    predicted_betas = predicted_betas.reshape(Y.shape[0],)
    indices = find_locations(predicted_betas, num_breaks)
    print(indices)
    visualize(X, betas, Y, indices)
    
