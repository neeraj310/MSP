# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from timeit import default_timer as timer
from typing import List
import sys

import numpy as np
sys.path.append('')
import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.queries import Query


class PointQuery(Query):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__(models)

    def predict(self, model_idx: int, key: int):
        return self.models[model_idx].predict(key)
    
    def predict_range_query(self, model_idx: int, query_l, query_u):
        print('Get keys in range (%d, %d), (%d, %d)' %(query_l[0], query_l[1], query_u[0], query_u[1] ))
        return self.models[model_idx].predict_range(query_l, query_u)

    def evaluate(self, test_data):
        data_size = test_data.shape[0]
        print("[Point Query] Evaluating {} datapoints".format(data_size))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            ys = []
            start_time = timer()
            for i in range(data_size):
                y = self.predict(idx, test_data.iloc[i, :-1])
                y = int(y // model.page_size)
                ys.append(y)
            end_time = timer()
            yhat = np.array(ys).reshape(-1, 1)
            ytrue = np.array(test_data.iloc[:, -1:])
            mse = metrics.mean_squared_error(yhat, ytrue)
            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)
        return mses, build_times

    def evaluate_range_query(self, test_range_query):
        data_size = test_range_query.shape[0]
        print("[Point Query] Evaluating {} datapoints".format(data_size))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
      
            start_time = timer()
            y_pred = self.predict_range_query(idx, test_range_query.iloc[0, :-1],test_range_query.iloc[-1, :-1])
            end_time = timer()
            if (y_pred.shape[0] != data_size):
                print( 'Nu of predicted entries in range query %d versus expected entries %d',y_pred.shape[0], data_size)
                mse = -1
            else:
                yhat = np.array(y_pred).reshape(-1, 1)
                ytrue =np.array(test_range_query.iloc[:, -1:])
                mse = metrics.mean_squared_error(yhat, ytrue)
            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                    model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)
            
        return mses, build_times
    def get_model(self, model_idx: int):
        return self.models[model_idx]
