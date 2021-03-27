# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List
import sys
sys.path.append('')
from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import split_train_test


class Query(object):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__()
        self.models = models

    def build(self, data, test_ratio):
        build_times = []
        x_train, _, x_test, y_test, x_index = split_train_test(
            data, test_ratio)
        for model in self.models:
            mse, build_time = model.train(x_train, x_index, x_test, y_test)
            print("{} model built in {:.4f} ms, mse={:4f}".format(
                model.name, build_time * 1000, mse))
            build_times.append(build_time)
        return build_times
