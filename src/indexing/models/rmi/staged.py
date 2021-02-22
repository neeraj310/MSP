'''
This file describes how staged model, i.e. recursive model works.

@author: Xiaozhe Yao
@updated: 22. Feb. 2021.
'''

from timeit import default_timer as timer
from typing import List
import numpy as np

from src.indexing.models import BaseModel
from src.indexing.models.ml.polynomial_regression import PolynomialRegression
from src.indexing.models.trees.b_tree import BTreeModel
from src.indexing.models.nn.fcn import FCNModel

class StagedModel(BaseModel):
    def __init__(self, model_strings) -> None:
        super().__init__("Staged Model")
        self.num_of_stages = len(model_strings)
        self.models = []
        for each in model_strings:
            if each == 'lr':
                self.models.append(PolynomialRegression(1))
            elif each == 'quadratic':
                self.models.append(PolynomialRegression(2))
            elif each == 'b-tree':
                self.models.append(BTreeModel(page_size=20))
            elif each == 'fcn':
                self.models.append(FCNModel)
            else:
                raise ValueError("Unsupported Model Type")

    def train(self, x_train, y_train, x_test, y_test):
        for each in self.models:
            pass