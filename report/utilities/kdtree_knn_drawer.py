# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
matplotlib.rcParams['text.usetex'] = True

class KDTree:
    """Simple KD tree class"""

    # class initialization function
    def __init__(self, data, mins, maxs):
        self.data = np.asarray(data)

        # data should be two-dimensional
        assert self.data.shape[1] == 2

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.child1 = None
        self.child2 = None

        if len(data) > 1:
            # sort on the dimension with the largest spread
            largest_dim = np.argmax(self.sizes)
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]

            # find split point
            N = self.data.shape[0]
            half_N = int(N / 2)
            split_point = 0.5 * (self.data[half_N, largest_dim]
                                 + self.data[half_N - 1, largest_dim])

            # create subnodes
            mins1 = self.mins.copy()
            mins1[largest_dim] = split_point
            maxs2 = self.maxs.copy()
            maxs2[largest_dim] = split_point

            # Recursively build a KD-tree on each sub-node
            self.child1 = KDTree(self.data[half_N:], mins1, self.maxs)
            self.child2 = KDTree(self.data[:half_N], self.mins, maxs2)

    def draw_rectangle(self, ax, depth=None):
        """Recursively plot a visualization of the KD tree region"""
        if depth == 0:
            rect = plt.Rectangle(self.mins, *self.sizes, ec='k', fc='none')
            ax.add_patch(rect)

        if self.child1 is not None:
            if depth is None:
                self.child1.draw_rectangle(ax)
                self.child2.draw_rectangle(ax)
            elif depth > 0:
                self.child1.draw_rectangle(ax, depth - 1)
                self.child2.draw_rectangle(ax, depth - 1)


#------------------------------------------------------------
# Create a set of structured random points in two dimensions

X=[[30,40],[5,25],[10,12],[70,70],[50,30],[35,45], [4,10]]

#------------------------------------------------------------
# Use our KD Tree class to recursively divide the space
KDT = KDTree(X, [0,0], [100, 100])

def draw_regions():
    #------------------------------------------------------------
    # Plot four different levels of the KD tree
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(wspace=0.1, hspace=0.15,
                        left=0.1, right=0.9,
                        bottom=0.05, top=0.9)

    for level in range(1, 5):
        ax = fig.add_subplot(2, 2, level, xticks=[], yticks=[])
        ax.scatter(X[:, 0], X[:, 1], s=9)
        KDT.draw_rectangle(ax, depth=level - 1)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title('level %i' % level)

    # suptitle() adds a title to the entire figure
    fig.suptitle('$k$d-tree Example')
    plt.show()

def draw_tree(bbox=False, knn_point=False):
    for each in X:
        plt.scatter(each[0], each[1])
        if each==[30,40]:
            plt.annotate(r"$({},{})$".format(each[0], each[1]), (each[0]-10, each[1]+1))
        elif each==[4,10]:
            plt.annotate(r"$({},{})$".format(each[0], each[1]), (each[0]-4, each[1]-5))
        else:
            plt.annotate(r"$({},{})$".format(each[0], each[1]), (each[0]+1, each[1]+1))
    plt.vlines(30, 0, 100)
    plt.vlines(50, 0, 70)
    plt.vlines(10, 0, 25)
    plt.hlines(25, 0, 30)
    plt.hlines(45, 30, 50)
    plt.hlines(70, 30, 100)
    plt.hlines(10, 0, 10)
    if bbox:
        rect = Rectangle((30,0),70, 70, linewidth=0, fill=None, hatch="\\")
        ax=plt.gca()
        ax.add_patch(rect)
        ax.annotate(r'Bounding Box of (50,30)', (70,0))
    if knn_point:
        plt.scatter(5, 35, marker='+')
        plt.annotate(r"$\mathcal{K}(5, 35, k)$", (3,40))
        x_values = [5, 30]
        y_values = [35, 35]
        plt.plot(x_values, y_values, color='green', linestyle='dashed')
        rect = Rectangle((0,0),30, 25, linewidth=0, fill=None, hatch="/")
        ax=plt.gca()
        ax.add_patch(rect)

    plt.axis('off')
    # plt.show()
    plt.savefig('./graphs/implementation/queries/knn_query_kdtree.pdf', transparent=True)

if __name__=="__main__":
    draw_tree(True, True)