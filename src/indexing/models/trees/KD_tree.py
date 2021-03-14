import csv
import heapq
import pickle
import random
from timeit import default_timer as timer

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KDTree


class KDTreeModel():
    def __init__(self):
        super(KDTreeModel, self).__init__()
        self.name = 'KD-Tree'
        self.kdtree = None

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
        mse = metrics.mean_absolute_error(y_test, y_predict_test)

        return mse, build_time

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
            for j, c in (
                (0, dx >= 0), (1, dx < 0)
            ):  # j is for left and right addition  and c is to check where to add
                if c and kd_node[j] is None:
                    kd_node[j] = [None, None, point]
                elif c:
                    self.add_point(point, dim, kd_node[j], i)

    def get_k_nearest(self,
                      point,
                      k,
                      dim,
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
                self.get_k_nearest(point, k, dim, kd_node[b], return_distances,
                                   i, heap)
        if is_root:
            neighbors = sorted((-h[0], h[1]) for h in heap)
            return neighbors if return_distances else [n[1] for n in neighbors]

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


#Below is the code to test the ground truth using sklearn KDtree
def sklearn_kdtree(points, dim):
    points = np.array(points)
    tree = KDTree(points, leaf_size=2)
    s = pickle.dumps(tree)
    pickle.loads(s)
    dist, _ = tree.query(test, k=3)
    return dist**2


# def sanity_check():
#     list_kdtree = []
#     for i in range(1,10):
#         dist_sklearn = sklearn_kdtree(points, k)
#         dist_kdtree = KDTreeModel.get_k_nearest(point, k, dim, kd_node[b],return_distances, i, heap)
#         for i in range(len(dist_kdtree)):
#             list_kdtree.append(dist_kdtree[i][0])
#         mse = metrics.mean_squared_error(np.array(list_kdtree), np.array(dist_sklearn))
#         i = i+1
"""
Below is all the testing code
"""
if __name__ == "__main__":

    # filename=sys.argv[1]
    # dim = sys.argv[2]

    filename = 'data/2d_lognormal_lognormal_1000000.csv'
    dim = 2

    # ***** Reading data points from csv ******
    points = []
    with open(filename, 'r') as csvfile:
        points_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(points_reader)
        for point in points_reader:
            points.append(list(np.float_(point)))

    test = [[3, 1]]
    result = []

    kdtree = KDTreeModel()
    bt = kdtree.build_kd_tree(points, dim)

    print(bt, " build time for kd")

    # # t_start = timer()
    # for t in test:
    #     result.append(tuple(kdtree.get_k_nearest(t, 2, dim)))
    # # t_end = timer()

    # dis_grnd_truth = sklearn_kdtree(points, dim)
    # print(result, "result")
    # # print(t_end-t_start, "Time taken")
    # print(dis_grnd_truth, "dis_grnd_truth")

# # Makes the KD-Tree for fast lookup
# def make_kd_tree(points, dim, i=0):
#     if len(points) > 1:
#         points.sort(key=lambda x: x[i])
#         i = (i + 1) % dim
#         half = len(points) >> 1
#         return [
#             make_kd_tree(points[: half], dim, i),
#             make_kd_tree(points[half + 1:], dim, i),
#             points[half]
#         ]
#     elif len(points) == 1:
#         return [None, None, points[0]]

# # Adds a point to the kd-tree
# def add_point(kd_node, point, dim, i=0):
#     if kd_node is not None:
#         dx = kd_node[2][i] - point[i]
#         i = (i + 1) % dim # i is to alternate bw x and y
#         for j, c in ((0, dx >= 0), (1, dx < 0)): # j is for left and right addition  and c is to check where to add
#             if c and kd_node[j] is None:
#                 kd_node[j] = [None, None, point]
#             elif c:
#                 add_point(kd_node[j], point, dim, i)

# # k nearest neighbors
# def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
#     import heapq
#     is_root = not heap
#     if is_root:
#         heap = []
#     if kd_node is not None:
#         dist = dist_func(point, kd_node[2])
#         dx = kd_node[2][i] - point[i]
#         if len(heap) < k:
#             heapq.heappush(heap, (-dist, kd_node[2]))
#         elif dist < -heap[0][0]:
#             heapq.heappushpop(heap, (-dist, kd_node[2]))
#         i = (i + 1) % dim
#         # Goes into the left branch, and then the right branch if needed
#         for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]): # dx*dx is r' and decide whether to check other branch or not.
#             get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
#     if is_root:
#         neighbors = sorted((-h[0], h[1]) for h in heap)
#         return neighbors if return_distances else [n[1] for n in neighbors]

# # For the closest neighbor
# def get_nearest(kd_node, point, dim, dist_func, return_distances=True, i=0, best=None):
#     if kd_node is not None:
#         dist = dist_func(point, kd_node[2])
#         dx = kd_node[2][i] - point[i]
#         if not best:
#             best = [dist, kd_node[2]]
#         elif dist < best[0]:
#             best[0], best[1] = dist, kd_node[2]
#         i = (i + 1) % dim
#         # Goes into the left branch, and then the right branch if needed
#         for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
#             get_nearest(kd_node[b], point, dim, dist_func, return_distances, i, best)
#     return best if return_distances else best[1]

# def puts(l):
#     for x in l:
#         print(x)

# def get_knn_naive(points, point, k, dist_func, return_distances=True):
#     neighbors = []
#     for i, pp in enumerate(points):
#         dist = dist_func(point, pp)
#         neighbors.append((dist, pp))
#     neighbors = sorted(neighbors)[:k]
#     return neighbors if return_distances else [n[1] for n in neighbors]

# def bench1():
#     kd_tree = make_kd_tree(points, dim)
#     for point in additional_points:
#         add_point(kd_tree, point, dim)
#     # result1.append(tuple(get_knn(kd_tree, [0] * dim, 1, dim, dist_sq_dim)))
#     for t in test:
#         result1.append(tuple(get_knn(kd_tree, t, 1, dim, dist_sq_dim)))

# def bench2():
#     all_points = points + additional_points
#     result2.append(tuple(get_knn_naive(all_points, [0] * dim, 8, dist_sq_dim)))
#     for t in test:
#         result2.append(tuple(get_knn_naive(all_points, t, 8, dist_sq_dim)))

# cProfile.run("bench1()")
# cProfile.run("bench2()")

# puts(result1[0])
# print("")
# puts(result2[0])
# print("")

# print("Is the result same as naive version?: {}".format(result1 == result2))

# print("")
# kd_tree = make_kd_tree(points, dim)

# print(get_nearest(kd_tree, [0] * dim, dim, dist_sq_dim))
