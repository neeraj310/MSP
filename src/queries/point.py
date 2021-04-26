# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from src.indexing.utilities.dataloaders import uniform_sample
import sys
from timeit import default_timer as timer
from typing import List

import numpy as np

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.queries import Query

sys.path.append('')

class PointQuery(Query):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__(models)
        self.debug_print = False

    def predict(self, model_idx: int, key: int):
        return self.models[model_idx].predict(key)

    def predict_range_query(self, model_idx: int, query_l, query_u):
        if self.debug_print:
            print('Get keys in range (%d, %d), (%d, %d)' %
                  (query_l[0], query_l[1], query_u[0], query_u[1]))
        return self.models[model_idx].predict_range_query(query_l, query_u)

    def predict_knn_query(self, model_idx: int, query, k):

        return self.models[model_idx].predict_knn_query(query, k)

    def evaluate(self, test_data):
        return self.evaluate_point(test_data)

    def evaluate_point(self, test_data):
        data_size = test_data.shape[0]
        if self.debug_print:
            print("[Point Query] Evaluating {} datapoints".format(data_size))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            ys = []
            start_time = timer()
            for i in range(data_size):
                y = self.predict(idx, test_data.iloc[i, :-1])
                y = int(y // model.page_size)
                if self.sample_ratio:
                    y = y / self.sample_ratio
                ys.append(y)
                # print("Evaluating {}/{}".format(i, data_size), end='\r')
            end_time = timer()
            yhat = np.array(ys).reshape(-1, 1)
            ytrue = np.array(test_data.iloc[:, -1:])
            mse = metrics.mean_squared_error(yhat, ytrue)
            mses.append(mse)
            if self.debug_print:
                print(
                    "{} model tested in {:.4f} seconds with mse {:.4f}".format(
                        model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)
        return mses, build_times

    def evaluate_range_query(self, test_range_query):
        data_size = np.array(test_range_query.shape[0])
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            if (model.name == 'Lisa Baseline'
                    or model.name == 'Scipy KD-Tree'):
                continue
            start_time = timer()
            y_pred = np.array(
                self.predict_range_query(idx, test_range_query.iloc[0, :-1],
                                         test_range_query.iloc[-1, :-1]))
            end_time = timer()
            if (y_pred.shape[0] != data_size):
                print(
                    'Num of predicted entries in range query %d versus expected entries %d',
                    y_pred.shape[0], data_size)
                mse = -1

            else:
                ytrue = np.array(test_range_query.iloc[:, -1:])
                mse = metrics.mean_squared_error(np.sort(y_pred),
                                                 np.sort(ytrue))

            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)

        return mses, build_times

    def evaluate_scipy_kdtree_knn_query(self, query, k):

        print("Get %d nearest neighbours for query %d %d" %
              (k, query[0], query[1]))
        for idx, model in enumerate(self.models):
            if model.name == 'Scipy KD-Tree':
                y_pred = self.models[idx].predict_knn_query(query, k)
                print('Grounftruth for query %d %d for %d neighbours' %
                      (query[0], query[1], k))
                print(y_pred)
                return y_pred
            else:
                continue
        return -1

    def evaluate_knn_query(self, query, ytrue, k):

        if self.debug_print:
            print("[Point Query %d %d]  Evaluating %d neighbours" %
                  (query[0], query[1], k))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            # if (model.name == 'Scipy KD-Tree') or (model.name == 'Lisa Baseline'):
            #     continue
            start_time = timer()
            y_pred = np.array(self.predict_knn_query(idx, query, k))
            end_time = timer()
            ytrue = np.squeeze(ytrue)
            if (y_pred.shape[0] != np.squeeze(ytrue).shape[0]):
                print(
                    f'Num of predicted entries in knn query {np.squeeze(y_pred).shape[0]} versus expected {ytrue.shape[0]} entries'
                )
                mse = -1

            else:
                yhat = np.array(y_pred).reshape(-1, 1)
                mse = metrics.mean_squared_error(yhat, ytrue)

                if (mse != 0):
                    print(yhat)
                    print('\n\n\n\n')
                    print(ytrue)
                    for i in range(ytrue.shape[0]):
                        if (yhat[i] != ytrue[i]):
                            print(' Predicted y %d Expected y %d' %
                                  (yhat[i], ytrue[i]))

            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)

        return mses, build_times

    def get_model(self, model_idx: int):
        return self.models[model_idx]
