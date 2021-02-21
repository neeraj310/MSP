from timeit import default_timer as timer
from typing import List

from sklearn import metrics

from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import split_train_test


class PointQuery(object):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__()
        self.models = models

    def build(self, data, test_ratio):
        build_times = []
        x_train, y_train, x_test, y_test = split_train_test(data, test_ratio)
        for model in self.models:
            mse, build_time = model.train(x_train, y_train, x_test, y_test)
            print("{} model built in {} seconds, mse={}".format(
                model.name, build_time, mse))
            build_times.append(build_time)
        return build_times

    def predict(self, model_idx, key):
        return self.models[model_idx].predict(key)

    def evaluate(self, test_data):
        data_size = test_data.shape[0]
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            ys = []
            start_time = timer()
            for i in range(data_size):
                y = self.predict(idx, test_data.iloc[i, 0])
                ys.append(y)
            end_time = timer()
            mse = metrics.mean_squared_error(test_data.iloc[:, 1], ys)
            mses.append(mse)
            print("{} model tested in {} seconds".format(
                model.name, end_time - start_time))
            build_times.append(end_time - start_time)
        return mses, build_times

    def get_model(self, model_idx: int):
        return self.models[model_idx]
