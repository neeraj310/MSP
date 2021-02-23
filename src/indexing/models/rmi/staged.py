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

PAGE_SIZE=10

class StagedModel(BaseModel):
    def __init__(self, model_types, num_models) -> None:
        super().__init__("Staged Model")
        self.num_of_stages = len(model_types)
        self.num_of_models = num_models
        if not self.num_of_models == self.num_of_stages:
            raise ValueError("Length of num_models is expected to be equal to len(model_types)")
        self.models:List = []
        self.model_types = model_types
    
    def _build_single_model(self,model_type, train_data):
        x_train = train_data[0]
        y_train = train_data[1]
        if model_type == 'lr':
            model = PolynomialRegression(1)
        elif model_type == 'quadratic':
            model = PolynomialRegression(2)
        elif model_type == 'b-tree':
            model = BTreeModel(page_size=PAGE_SIZE)
        elif model_type == 'fcn':
            model = FCNModel()
        else:
            raise ValueError("Unsupported Model Type")
        model.fit(x_train, y_train)
        return model

    def train(self, x_train, y_train, x_test, y_test):
        train_data = (x_train, y_train)
        # a 2-d array indexed by [stage][model_id]
        train_datas = [[train_data]]
        for stage in range(self.num_of_stages):
            self.models.append([])
            for model_id in range(self.num_of_models[stage]):
                model = self._build_single_model(self.model_types[stage], train_datas[stage][model_id])
                self.models[stage].append(model)
            if not stage == self.num_of_stages-1:
                # if it is not the last stage,
                # prepare dataset for the next stage
                next_xs = [[] for i in range(self.num_of_models[stage+1])]