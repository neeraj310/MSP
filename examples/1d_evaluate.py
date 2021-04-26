# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import uuid
from typing import List

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.indexing.models import BaseModel
from src.indexing.models.ml.polynomial_regression import PRModel
from src.indexing.models.nn.fcn import FCNModel
# from src.indexing.models.nn.conv import ConvModel
from src.indexing.models.nn.pwlf_conv import ConvModel
from src.indexing.models.rmi.staged import StagedModel
from src.indexing.models.trees.b_tree import BTree, BTreeModel
from src.indexing.utilities.dataloaders import uniform_sample
from src.indexing.utilities.metrics import get_memory_size
from src.indexing.utilities.results import write_results
from src.queries.point import PointQuery

TEST_RATIO = 0.2
B_TREE_DEGREE = 20


def load_data(filename, sample_size=None):
    data = pd.read_csv(filename)
    train_data = data
    test_data = data.sample(n=int(TEST_RATIO * len(data)))
    if sample_size is not None:
        train_data = uniform_sample(data, sample_size)
    page_size = len(data) // np.max(data.iloc[:, 1])
    return train_data, test_data, data, page_size


def predict_all(models, data, experiment_id):
    x = data.iloc[:, :-1].values
    x = x.reshape(-1)
    gt_y = data.iloc[:, 1].values
    pred_ys = []
    for model in models:
        pred_y = []
        for each in x:
            pred_y.append(int(model.predict(each) // model.page_size))
        if model.name == 'Fully Connected Neural Network':
            # print(pred_y)
            pass
        pred_ys.append(pred_y)
    results = {}
    results['x'] = x
    results['ground_truth'] = gt_y
    for idx, model in enumerate(models):
        results[model.name] = pred_ys[idx]
    df = pd.DataFrame.from_dict(results)
    df.to_csv('result_{}.csv'.format(experiment_id), index=False)
    print("Results of experiment {} have been saved to result.csv".format(experiment_id))


def train(filename, settings={}):
    experiment_id = uuid.uuid4().__str__()[0:8]
    train_data, test_data, data, page_size = load_data(
        filename, settings['sample_size'])
    models = []
    sample_ratio = len(train_data)/len(data)
    if settings['conv']:
        model = ConvModel(page_size=page_size, num_breaks=128)
        models.append(model)
    if settings['b-tree']:
        model = BTreeModel(page_size, B_TREE_DEGREE)
        models.append(model)
    if settings['fcn']:
        model = FCNModel(page_size=page_size, layers=[1, 9, 9, 1], activations=[
                         'relu', 'relu', 'relu'], epochs=10000)
        models.append(model)
    if settings['staged']:
        model = StagedModel(['fcn', 'fcn', 'fcn'], [1, 200, 4000], page_size)
        models.append(model)
    if len(models) == 0:
        raise ValueError("There must be at least one model!")
    ptq = PointQuery(models)
    build_times = ptq.build(train_data, TEST_RATIO,
                            use_index=True, sample_ratio=sample_ratio)
    mses, eval_times = ptq.evaluate(data)
    result = []
    header = [
        "Name", "Build Time (s)", "Evaluation Time (s)", "Evaluation Error (MSE)", "Memory Size (KiB)"
    ]
    for index, model in enumerate(models):
        result.append([
            model.name, build_times[index], eval_times[index], mses[index], get_memory_size(
                model)
        ])
    if settings['write']:
        write_results(experiment_id, result)
    print("Experiment ID: {}".format(experiment_id))
    print(tabulate(result, header))
    if settings['draw_curve']:
        predict_all(models, data, experiment_id)


if __name__ == "__main__":
    settings = {
        "b-tree": True,
        "fcn": True,
        "staged": True,
        "conv": True,
        "write": True,
        "draw_curve": True,
        "sample_size": None,
    }
    filename = sys.argv[1]
    print(settings)
    train(filename, settings)
