# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from src.indexing.utilities.metrics import get_memory_size

import numpy as np
from scipy.spatial import KDTree

ns = [10,100,1000]
for n in ns:
    x, y = np.mgrid[0:n, 0:n]
    tree = KDTree(np.c_[x.ravel(), y.ravel()])

    print("Memory Size for n={}: {}".format(n*n, get_memory_size(tree)))
