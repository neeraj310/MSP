import sys

import pandas as pd
from tabulate import tabulate

from src.indexing.models.trees.b_tree import BTreeModel
from src.indexing.models.ml.linear_regression import LRModel
from src.queries.point import PointQuery

ratio = 0.2
b_tree_page_size = 20


def load_1D_Data(filename):
    data = pd.read_csv(filename)
    test_data = data.sample(n=int(ratio * len(data)))
    return data, test_data


def evaluate(filename):
    data, test_data = load_1D_Data(filename)
    btm = BTreeModel(b_tree_page_size)
    lrm = LRModel()
    models = [btm, lrm]
    ptq = PointQuery(models)
    build_times = ptq.build(data, ratio)
    mses, eval_times = ptq.evaluate(test_data)
    result = []
    header = [
        "Name", "Build Time (s)", "Evaluation Time (s)", "Evaluation Error (MSE)"]
    for index, model in enumerate(models):
        result.append([model.name, build_times[index],
                       eval_times[index], mses[index]])
    print(tabulate(result, header))


if __name__ == "__main__":
    filename = sys.argv[1]
    evaluate(filename)
