import sys
import pandas as pd
import numpy as np
CELL_SIZE = 10


def grid_cell(data):
    pass


def mapping():
    pass


def predict_shard():
    pass


def evaluate(filename):
    data = pd.read_csv(filename)
    x_range = np.max(data.x)
    y_range = np.max(data.y)
    num_of_keys = len(data)
    keys_per_cell = num_of_keys // (CELL_SIZE * CELL_SIZE)


if __name__ == "__main__":
    filename = sys.argv[1]
    evaluate(filename)
