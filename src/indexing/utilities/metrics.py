# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
from pympler import asizeof


def mean_squared_error(yhat, y):
    if not isinstance(yhat, np.ndarray):
        yhat = np.array(yhat)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    # return (np.square(yhat.reshape(-1) - y.reshape(-1))).mean()
    yhat_squeezed = np.squeeze(yhat)
    yhat_list = []
    if yhat.shape[0] != np.squeeze(y).shape[0]:

        for i in range(1,yhat_squeezed.shape[0],2):
            yhat_list.append(yhat_squeezed[i][2])
        result = np.square(np.array(yhat_list) - np.squeeze(y)).mean()
    else:
        result = np.square(yhat_squeezed - np.squeeze(y)).mean()

    return result


def get_memory_size(obj):
    '''
    return the memory size in kilo bytes
    '''
    return asizeof.asizeof(obj) / 1024
