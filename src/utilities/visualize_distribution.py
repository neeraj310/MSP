import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.indexing.utilities.dataloaders import uniform_sample


def visualize(dist_name, filename):
    data = pd.read_csv(filename)
    data = uniform_sample(data, size=1000)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values
    plt.scatter(x, y, s=np.pi * 3, alpha=0.5)
    plt.title('x-y of {} distribution'.format(dist_name))
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.show()
    plt.savefig('{}.pdf'.format(dist_name))


if __name__ == "__main__":
    filename = sys.argv[1]
    distribution = filename.split("_")[1]
    visualize(distribution, filename)
