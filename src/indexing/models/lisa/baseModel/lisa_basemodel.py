# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:36:30 2021

@author: neera
"""

from merit.models.trees.item import Item
from typing import Tuple
from merit.models.base.model import BaseModel
from timeit import default_timer as timer
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

class LisaBaseModel():
    def __init__(self, degree) -> None:
        self.pageCount  = degree
        self.denseArray = np.zeros((degree, 3))
        self.num_of_keys = 0
        self.KeysPerPage = 0
        print('In LisaBase Model ')
   

    def train(self, x_train, y_train, x_test, y_test):
        
        return 