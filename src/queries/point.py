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

    def get_model(self, model_idx: int):
        return self.models[model_idx]
