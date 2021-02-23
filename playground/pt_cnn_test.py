# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn
import torch.nn.functional as F
import tinyml
from tinyml.layers import Conv2D

def pt_conv(x, w):
    x = torch.from_numpy(x).float()
    w = torch.from_numpy(w).float()
    return F.conv1d(x, w)

def tinyml_conv(x, w):
    layer = Conv2D()