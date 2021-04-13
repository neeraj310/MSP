import csv
import heapq
import pickle
import random
import sys
import numpy as np

from timeit import default_timer as timer
from sklearn.neighbors import KDTree
from collections import Sequence
from itertools import chain, count


sys.path.append('')
from src.indexing.utilities.metrics import mean_squared_error
from src.indexing.utilities.metrics import get_memory_size


class KDTreeModel():
    def __init__(self):
        super(KDTreeModel, self).__init__()
        self.name = 'KD-Tree'
        self.kdtree = None
        self.page_size = 1

    def train(self, x_train, y_train, x_test, y_test, dim=2):

        #Build kd tree with train data
        data_train = np.hstack((x_train, y_train))
        data_train = data_train.tolist()
        build_time = self.build_kd_tree(data_train)

        # search points kd tree with test data
        y_predict_test = []
        for key in x_test:
            nearest = self.get_nearest(key, dim=2)
            y_predict_test.append(nearest[1][-1])

        y_predict_test = np.array(y_predict_test)
        mse = mean_squared_error(y_test, y_predict_test)

        return mse, build_time
    
    def predict_range_query(self,query_l, query_u,kd_node='init',i=0,out=None):
        xmin = query_l[0]
        xmax = query_u[0]
        ymin = query_l[1]
        ymax = query_u[1]
        self.query_l = query_l
        self.query_u = query_u

        if out==None:
            out = []

        if kd_node=='init':
            kd_node = self.kdtree

        if kd_node is not None:
            # xmin,ymin,xmax,ymax=area
            area = xmin,ymin,xmax,ymax
            # acceptance of point within range
            if kd_node[2][0]>=xmin and kd_node[2][0]<=xmax and kd_node[2][1]>=ymin and kd_node[2][1]<=ymax:
                out.append(kd_node[2][2])

            #for traversing left
            if (kd_node[2][i]>= xmin and i==0) or (kd_node[2][i]>= ymin and i==1):
                self.predict_range_query(self.query_l, self.query_u,kd_node[0],(i+1)%2,out)

            #for traversing right
            if (kd_node[2][i]<= xmax and i==0) or (kd_node[2][i]<= ymax and i==1):
                self.predict_range_query(self.query_l, self.query_u,kd_node[1],(i+1)%2,out)
        
        return out
    def depth(self, tree):
        tree_init = tree
        tree = iter(tree)
        try:
            for level in count():
                tree = chain([next(tree)], tree)
                tree = chain.from_iterable(s for s in tree if isinstance(s, Sequence))
        except StopIteration:
            return level, get_memory_size(tree_init)

    def predict(self, key):
        nearest = self.get_nearest(key, dim=2)
        return nearest[1][-1]

    def build_kd_tree(self, points, dim=2):
        start_time = timer()
        self.kdtree = self.build(points, dim)
        end_time = timer()       
        build_time = end_time - start_time
        return build_time

    def build(self, points, dim, i=0):
        if len(points) > 1:
            points.sort(key=lambda x: x[i])
            i = (i + 1) % dim
            half = len(points) >> 1
            return [
                self.build(points[:half], dim, i),
                self.build(points[half + 1:], dim, i), points[half]
            ]
        elif len(points) == 1:
            return [None, None, points[0]]

    def add_point(self, point, dim, kd_node=None, i=0):

        if kd_node is None:
            kd_node = self.kdtree
        if kd_node is not None:
            dx = kd_node[2][i] - point[i]
            i = (i + 1) % dim  # i is to alternate bw x and y
            for j, c in ((0, dx >= 0), (1, dx < 0)):  # j is for left and right addition  and c is to check where to add
                if c and kd_node[j] is None:
                    kd_node[j] = [None, None, point]
                elif c:
                    self.add_point(point, dim, kd_node[j], i)


    def predict_knn_query_find(self,
                      point,
                      k,
                      dim=2,
                      kd_node='init',
                      return_distances=True,
                      i=0,
                      heap=None):

        if kd_node == 'init':
            kd_node = self.kdtree

        is_root = not heap
        if is_root:
            heap = []
        if kd_node is not None:
            dist = self.dist_sq_dim(point, kd_node[2], dim)
            dx = kd_node[2][i] - point[i]
            if len(heap) < k:
                heapq.heappush(heap, (-dist, kd_node[2]))
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, (-dist, kd_node[2]))
            i = (i + 1) % dim
            # Goes into the left branch, and then the right branch if needed
            for b in [dx < 0] + [dx >= 0] * (
                    dx * dx < -heap[0][0]
            ):  # dx*dx is r' and decide whether to check other branch or not.
                self.predict_knn_query_find(point, k, dim, kd_node[b], return_distances,
                                   i, heap)
        if is_root:
            neighbors = sorted((-h[0], h[1]) for h in heap)
            return neighbors if return_distances else [n[1] for n in neighbors]

    def predict_knn_query(self,
                      point,
                      k,
                      dim=2,
                      kd_node='init',
                      return_distances=True,
                      i=0,
                      heap=None):

        y_pred = self.predict_knn_query_find(point,k,dim,kd_node,return_distances,i,heap)
        y_pred = np.array(y_pred)
        # yhat=np.vstack(y_pred[:,1])[:,2]
        yhat=np.vstack(y_pred[:,0])
        return np.sqrt(yhat)
                      


    def get_nearest(self,
                    point,
                    dim,
                    kd_node='init',
                    return_distances=True,
                    i=0,
                    nearest=None):

        if kd_node == 'init':
            kd_node = self.kdtree

        if kd_node is not None:
            dist = self.dist_sq_dim(point, kd_node[2], dim)
            dx = kd_node[2][i] - point[i]
            if not nearest:
                nearest = [dist, kd_node[2]]
            elif dist < nearest[0]:
                nearest[0], nearest[1] = dist, kd_node[2]
            i = (i + 1) % dim
            # Goes into the left branch, and then the right branch if needed
            for b in [dx < 0] + [dx >= 0] * (dx * dx < nearest[0]):
                self.get_nearest(point, dim, kd_node[b], return_distances, i,
                                 nearest)
        return nearest if return_distances else nearest[1]

    def rand_point(self, dim):
        return [random.uniform(-1, 1) for d in range(dim)]

    def dist_sq(self, a, b, dim):
        return sum((a[i] - b[i])**2 for i in range(dim))

    def dist_sq_dim(self, a, b, dim):
        return self.dist_sq(a, b, dim)

    def get_storage(self):

        return self.kdtree

"""
Below is all the testing code
"""
if __name__ == "__main__":

    # filename=sys.argv[1]
    # dim = sys.argv[2]
    m = 100
    # for i in range(0,4):
    #     m = m*10
    #     filename = 'data/2d_lognormal_lognormal_' + str(m) + '.csv'
    #     print(filename)
    #     dim = 2

    # ***** Reading data points from csv ******
    filename = 'data/2d_lognormal_lognormal_' + str(19000000) + '.csv'
    dim =2
    points = []
    with open(filename, 'r') as csvfile:
        points_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(points_reader)
        for point in points_reader:
            points.append(list(np.float_(point)))
    # points = [[11,12],[3,4],[4,5],[6,7],[8,9]]
    test = [[3, 1]]
    result = []

    kdtree = KDTreeModel()
    bt = kdtree.build_kd_tree(points, dim)

    levels, storage = kdtree.depth(kdtree.kdtree)

    print(levels, "levels of KDTree")

    print(storage, "Storage of KDTree")

    

