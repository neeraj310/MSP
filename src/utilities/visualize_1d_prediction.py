import sys

import pandas as pd
import matplotlib.pyplot as plt

def draw(filename):
    data = pd.read_csv(filename)
    x = data.iloc[:, 0]
    gt = data.iloc[:, 1]
    plt.scatter(x, gt, linewidths=0.01)
    total_columns = len(data.columns)
    for i in range(total_columns-2):
        predictions = data.iloc[:, i+2]
        plt.scatter(x, predictions, linewidth=0.01)
    plt.title('ReLU Activation')
    plt.savefig('relu.pdf')
if __name__=="__main__":
    filename = sys.argv[1]
    draw(filename)