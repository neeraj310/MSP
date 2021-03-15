# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
from pympler import asizeof

def mse(yhat, y):
    return (np.square(yhat - y)).mean()

def get_memory_size(obj):
    '''
    return the memory size in kilo bytes
    '''
    return asizeof.asizeof(obj)/1024