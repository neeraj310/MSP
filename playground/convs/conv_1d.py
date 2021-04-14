# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from numpy.lib.npyio import load
import pandas as pd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def generate_x_y(number_of_pairs):
    from numpy.random import default_rng
    rng = default_rng()
    vals = rng.standard_normal(number_of_pairs)
    vals = np.abs(vals) * number_of_pairs / 10
    x = np.sort(vals)
    y = np.linspace(0, number_of_pairs - 1, number_of_pairs)
    return x, y

def load_1D_Data(filename):
    data = pd.read_csv(filename)
    return data

def np_conv(x, w):
    output = np.convolve(x, w, 'same')
    return output.astype(np.float)

def vis(x, y, conv):
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, conv, '+')
    plt.show()

def filter_conv(threshold, conv_out):
    conv_out = np.where(
        np.abs(conv_out) > threshold,
        conv_out,
        0
    )
    return conv_out

def train_piecewise():
    pass

if __name__=="__main__":
    # import sys
    # filename = sys.argv[1]
    # data = load_1D_Data(filename)
    w = np.array([-1,2,-1])
    x, y = generate_x_y(10)
    conv_out = np.abs(np_conv(x, w))
    threshold = 0
    conv_out = filter_conv(threshold, conv_out)
    print(conv_out)
    vis(x, y, conv_out)
