from timeit import default_timer as timer
from typing import List

from sklearn import metrics

from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import split_train_test


class Query(object):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__()
        self.models = models
    
    def build(self, data, test_ratio):
        build_times = []
        x_train, y_train, x_test, y_test = split_train_test(data, test_ratio)
        for model in self.models:
            mse, build_time = model.train(x_train, y_train, x_test, y_test)
            print("{} model built in {:.4f} ms, mse={:4f}".format(
                model.name, build_time * 1000, mse))
            build_times.append(build_time)
        return build_times