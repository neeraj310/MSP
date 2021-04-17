# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
!!One Dimensional Only
This file reads several distributions, normalize them, and then concat them into a long array
'''
import csv
import os
import sys

import numpy as np
import pandas as pd

from src.indexing.utilities.dataloaders import normalize


def prepare(filenames):
    X = np.array([])
    Y = np.array([])
    for each in filenames:
        data = pd.read_csv(each)
        x = data.iloc[:, :-1].values
        y = data.iloc[:, 1].values
        X = np.append(X, normalize(x))
        Y = np.append(Y, normalize(y))
    return X, Y


def write_to_csv(X, Y):
    data_path = os.path.join("data", "1d_merged.csv")
    with open(data_path, "w+") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])
        for index, number in enumerate(X):
            csv_writer.writerow([number, Y[index]])


if __name__ == "__main__":
    files = [x for x in sys.argv[1:]]
    X, Y = prepare(files)
    write_to_csv(X, Y)
