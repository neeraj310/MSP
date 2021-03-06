# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
from typing import List

import pandas as pd
from tabulate import tabulate

from src.indexing.models import BaseModel
from src.indexing.models.ml.polynomial_regression import PRModel
from src.indexing.models.nn.fcn import FCNModel
from src.indexing.models.rmi.staged import StagedModel
from src.indexing.models.trees.b_tree import BTreeModel
from src.indexing.models.lisa.basemodel import LisaBaseModel
from src.queries.point import PointQuery

from sklearn import metrics
import numpy as np

ratio = 0.2



def load_2D_Data(filename):
    data = pd.read_csv(filename)
    #data = data[0:100]
    test_data = data.sample(n=int(ratio * len(data)))
    return data, test_data


def evaluate(filename):
    data, test_data = load_2D_Data(filename)
    lisaBm = LisaBaseModel(100)
    
    '''
    btm = BTreeModel(b_tree_page_size)
    fcn = FCNModel()

    lrm = PRModel(1)
    prm = PRModel(2)
    sgm = StagedModel(['lr', 'b-tree', 'lr'], [1, 2, 8])
    models = [btm, lrm, prm, sgm]
    '''
    models = [lisaBm]
    ptq = PointQuery(models)
    build_times = ptq.build(data, 0.00002)
    
    i = 10
    result = []
    header = [
        "Name", "Test Data Size","Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)","Evaluation Error (MSE)"
        ]
    while (i <= 1000000):
        mses, eval_times = ptq.evaluate(test_data.iloc[:i, :])
        
        for index, model in enumerate(models):
            result.append(
                [model.name, i, build_times[index], eval_times[index], 
                eval_times[index]/i, mses[index]])
        print(len(result))    
        i = i*10
    print(tabulate(result, header))
    #models_predict(data, models)
    

def models_predict(data, models: List[BaseModel]):
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
    results['x1'] = x[:,0]
    results['x2'] = x[:,1]
    results['ground_truth'] = gt_y
   
    for idx, model in enumerate(models):
        results[model.name] = pred_ys[idx]
        print('mse error for model %s is %f' %(model.name,metrics.mean_squared_error(np.array(pred_ys[idx]), gt_y)))
        
    df = pd.DataFrame.from_dict(results)
    df.to_csv('result.csv', index=False)
    print("Results have been saved to result.csv")


if __name__ == "__main__":
    filename = sys.argv[1]
    evaluate(filename)
