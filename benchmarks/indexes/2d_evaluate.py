# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
from typing import List

import pandas as pd
from tabulate import tabulate

sys.path.append('')
import numpy as np

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
# from src.indexing.models.lisa.basemodel import LisaBaseModel
# from src.indexing.models.lisa.lisa import LisaModel
from src.indexing.models.trees.KD_tree import KDTreeModel
from src.indexing.models.trees.scipykdtree import ScipyKDTreeModel
from src.queries.point import PointQuery
from src.queries.range import RangeQuery

ratio = 0.5


def load_2D_Data(filename):
    data = pd.read_csv(filename)
    #data = data[0:100]
    # Remove duplicates
    col_names = list(data.columns)[:-1]
    data.drop_duplicates(subset=col_names, ignore_index=True, inplace=True)
    data = data[0:10000]
    test_data = data  #data.sample(n=int(ratio * len(data)))
    return data, test_data


def create_models(filename):
    data, test_data = load_2D_Data(filename)
    # LisaBaseModel(100)
    kdtree = KDTreeModel()
    scipykdtree = ScipyKDTreeModel(leafsize=10)
    # lisa = LisaModel(cellSize=4, nuOfShards=5)

    models = [kdtree, scipykdtree]
    ptq = PointQuery(models)
    build_times = ptq.build(data, 0.002, use_index=False)
    return (models, ptq, test_data, build_times)


def point_query_eval(models, ptq, test_data, build_times):
    i = 1000
    result = []
    header = [
        "Name", "Test Data Size", "Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)", "Evaluation Error (MSE)"
    ]
    # while (i <= 10000):
    #Sample a point
    mses, eval_times = ptq.evaluate_point(test_data.iloc[:i, :])

    for index, model in enumerate(models):
        result.append([
            model.name, i, build_times[index], eval_times[index],
            eval_times[index] / i, mses[index]
        ])
    print(len(result))

    print(tabulate(result, header))


def range_query_eval(models, ptq, test_data, build_times):
    i = 10

    header = [
        "Name", "Query Size", "Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)", "Evaluation Error (MSE)"
    ]
    print(test_data.size)
    print(test_data.shape)

    range_size_list = [10, 100, 1000, 10000]
    averag_eval_time_across_ranges = np.zeros(len(models))
    average_loop_size = 20
    for j in range_size_list:
        i = 0
        result = []
        total_eval_time_per_range_size = np.zeros(len(models))
        while (i <= average_loop_size):
            random_point_idx = test_data.shape[0] - j
            if (random_point_idx != 0):
                idx = np.random.randint(test_data.shape[0] - j)
            else:
                idx = 0
            if (idx + j) > test_data.shape[0]:
                break
            test_range_query = test_data.iloc[idx:idx + j, :]
            mses, eval_times = ptq.evaluate_range_query(test_range_query)

            for index, model in enumerate(models):
                if (model.name == 'Scipy KD-Tree'):
                    continue
                result.append([
                    model.name, j, build_times[index], eval_times[index],
                    eval_times[index] / j, mses[index]
                ])
                total_eval_time_per_range_size[index] += (eval_times[index] /
                                                          j)

            i = i + 1
        print(tabulate(result, header))
        for index, model in enumerate(models):
            if (model.name == 'Scipy KD-Tree') or (model.name
                                                   == 'Lisa Baseline'):
                continue
            print('Average Eval time for model %s query size %d is %f' %
                  (model.name, j,
                   total_eval_time_per_range_size[index] / average_loop_size))
            averag_eval_time_across_ranges[
                index] += total_eval_time_per_range_size[
                    index] / average_loop_size
        #print(len(result))

    for index, model in enumerate(models):
        if (model.name == 'Scipy KD-Tree') or (model.name == 'Lisa Baseline'):
            continue
        print('average query time for model %s across different ranges %f' %
              (model.name,
               averag_eval_time_across_ranges[index] / len(range_size_list)))


def knn_query_eval(models, ptq, test_data, build_times):
    i = 3

    header = [
        "Name", "K Value", "Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)", "Evaluation Error (MSE)"
    ]
    k_list = [3, 4, 5, 6, 7, 10]
    print(test_data.size, "test data size")
    print(test_data.shape, " test data shape")
    averag_eval_time_across_knn_queries = np.zeros(len(models))
    for i in k_list:
        average_loop_size = 10
        result = []
        total_eval_time_per_k_size = np.zeros(len(models))
        for j in range(average_loop_size):
            idx = np.random.randint(test_data.shape[0] - 1000)

            if (idx + i) > test_data.shape[0]:
                break
            query = test_data.iloc[idx, 0:2]
            y_gt = ptq.evaluate_scipy_kdtree_knn_query(query, k=i)
            mses, eval_times = ptq.evaluate_knn_query(query, y_gt, k=i)

            for index, model in enumerate(models):
                # if (model.name == 'Scipy KD-Tree') or (model.name == 'Lisa Baseline') :
                # continue
                result.append([
                    model.name, i, build_times[index], eval_times[index],
                    eval_times[index] / i, mses[index]
                ])
                total_eval_time_per_k_size[index] += (eval_times[index] / i)
        print(tabulate(result, header))
        for index, model in enumerate(models):
            if (model.name == 'Scipy KD-Tree') or (model.name
                                                   == 'Lisa Baseline'):
                continue
            print(
                'Average Eval time for model %s for knn query with k = %d, is %f'
                % (model.name, i,
                   total_eval_time_per_k_size[index] / average_loop_size))
            averag_eval_time_across_knn_queries[
                index] += total_eval_time_per_k_size[index] / average_loop_size

        i = i + 1
    for index, model in enumerate(models):
        if (model.name == 'Scipy KD-Tree') or (model.name == 'Lisa Baseline'):
            continue
        print('average query time for model %s across different ranges %f' %
              (model.name,
               averag_eval_time_across_knn_queries[index] / len(k_list)))


def models_predict_point(data, models: List[BaseModel]):
    data = data.to_numpy()
    x = data[:, :-1]
    gt_y = data[:, -1:].reshape(-1)
    pred_ys = []
    for model in models:
        pred_y = []
        for each in x:
            pred_y.append(int(model.predict(each)))
        pred_ys.append(pred_y)
    results = {}
    results['x1'] = x[:, 0]
    results['x2'] = x[:, 1]
    results['ground_truth'] = gt_y

    for idx, model in enumerate(models):
        results[model.name] = pred_ys[idx]
        print('mse error for model %s is %f' %
              (model.name,
               metrics.mean_squared_error(np.array(pred_ys[idx]), gt_y)))

    df = pd.DataFrame.from_dict(results)
    df.to_csv('result.csv', index=False)
    print("Results have been saved to result.csv")


if __name__ == "__main__":

    # filename = sys.argv[1]
    filename = 'data/2d_lognormal_lognormal_1000000.csv'

    # evaluate(filename)
    (models, ptq, test_data, build_times) = create_models(filename)
    range_query_eval(models, ptq, test_data, build_times)
    knn_query_eval(models, ptq, test_data, build_times)
    point_query_eval(models, ptq, test_data, build_times)
