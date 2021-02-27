import csv
import os
import random
import sys

import numpy as np

DATA_SIZE = 10000
BLOCK_SIZE = 10
FACTOR = 10


def get_data(distribution, size):
    data = []
    if distribution == "UNIFORM":
        data = random.sample(range(size * FACTOR), size)
    elif distribution == "BINOMIAL":
        data = np.random.binomial(size / 2, 0.8, size)
    elif distribution == "POISSON":
        data = np.random.poisson(200, size)
    elif distribution == "EXPONENTIAL":
        data = np.random.exponential(150, size)
    elif distribution == "LOGNORMAL":
        data = np.random.lognormal(3, 2, size)
    else:
        data = np.random.normal(size, size * FACTOR, size)
    data.sort()
    data = data + abs(np.min(data))
    return data


def generate_1d_data(distribution, data_size=DATA_SIZE):
    data = get_data(distribution, data_size)
    multiplicant = 1
    if distribution == "EXPONENTIAL":
        multiplicant = 100
    elif distribution == "LOGNORMAL":
        multiplicant = 10000
    data_path = os.path.join(
        "data", "1d_" + distribution.lower() + "_" + str(data_size) + ".csv")
    with open(data_path, "w+") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["val", "block"])
        for index, number in enumerate(data):
            csv_writer.writerow(
                [int(number * multiplicant), index // BLOCK_SIZE])


if __name__ == "__main__":
    distribution = sys.argv[1]
    data_size = int(sys.argv[2])
    generate_1d_data(distribution.upper(), data_size)
