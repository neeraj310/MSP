import sys

import numpy as np
import pandas as pd

CELL_SIZE = 10


def grid_cell(data):
    pass


def mapping():
    pass


def predict_shard():
    pass


def evaluate(filename):
    data = pd.read_csv(filename)
    np.max(data.x)
    np.max(data.y)
    num_of_keys = len(data)
    num_of_keys // (CELL_SIZE * CELL_SIZE)


if __name__ == "__main__":
    filename = sys.argv[1]
    evaluate(filename)
