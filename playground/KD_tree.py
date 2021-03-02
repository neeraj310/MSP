
# Makes the KD-Tree for fast lookup
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]

# Adds a point to the kd-tree
def add_point(kd_node, point, dim, i=0):
    if kd_node is not None:
        dx = kd_node[2][i] - point[i]
        i = (i + 1) % dim # i is to alternate bw x and y
        for j, c in ((0, dx >= 0), (1, dx < 0)): # j is for left and right addition  and c is to check where to add
            if c and kd_node[j] is None:
                kd_node[j] = [None, None, point]
            elif c:
                add_point(kd_node[j], point, dim, i)

# k nearest neighbors
def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]): # dx*dx is r' and decide whether to check other branch or not.
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]

# For the closest neighbor
def get_nearest(kd_node, point, dim, dist_func, return_distances=True, i=0, best=None):
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if not best:
            best = [dist, kd_node[2]]
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
            get_nearest(kd_node[b], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]



"""
Below is all the testing code
"""

import random, cProfile


def puts(l):
    for x in l:
        print(x)


def get_knn_naive(points, point, k, dist_func, return_distances=True):
    neighbors = []
    for i, pp in enumerate(points):
        dist = dist_func(point, pp)
        neighbors.append((dist, pp))
    neighbors = sorted(neighbors)[:k]
    return neighbors if return_distances else [n[1] for n in neighbors]

dim = 2

def rand_point(dim):
    return [random.uniform(-1, 1) for d in range(dim)]

def dist_sq(a, b, dim):
    return sum((a[i] - b[i]) ** 2 for i in range(dim))

def dist_sq_dim(a, b):
    return dist_sq(a, b, dim)

# set_points= [[412,273],[364,293],[32,107]]

set_points= [[5,4]]
# points = [PointContainer(rand_point(dim)) for x in range(6)]
points = set_points
# additional_points = [PointContainer(rand_point(dim)) for x in range(50)]
# additional_points =[[429,195],[229,113]]
additional_points =[[2,6],[13,3],[8,7],[3,1],[10,2]]
#points = [rand_point(dim) for x in range(5000)]
# test = [rand_point(dim) for x in range(100)]
test = [[9,4]]
result1 = []
result2 = []


def bench1():
    kd_tree = make_kd_tree(points, dim)
    for point in additional_points:
        add_point(kd_tree, point, dim)
    # result1.append(tuple(get_knn(kd_tree, [0] * dim, 1, dim, dist_sq_dim)))
    for t in test:
        result1.append(tuple(get_knn(kd_tree, t, 1, dim, dist_sq_dim)))


def bench2():
    all_points = points + additional_points
    result2.append(tuple(get_knn_naive(all_points, [0] * dim, 8, dist_sq_dim)))
    for t in test:
        result2.append(tuple(get_knn_naive(all_points, t, 8, dist_sq_dim)))

cProfile.run("bench1()")
cProfile.run("bench2()")

puts(result1[0])
print("")
puts(result2[0])
print("")

print("Is the result same as naive version?: {}".format(result1 == result2))

print("")
kd_tree = make_kd_tree(points, dim)

print(get_nearest(kd_tree, [0] * dim, dim, dist_sq_dim))
