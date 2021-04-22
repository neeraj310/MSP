# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

plt.xlim([0, 2])
plt.ylim([0, 2])

points=[(0.7, 0.5), (0.7, 1.5), (1.5, 0.5)]

plt.scatter([x[0] for x in points], [x[1] for x in points])

#plt.vlines(1, 0, 2, linestyles="dotted")
#plt.hlines(1, 0, 2, linestyles="dotted")
ax = plt.gca()
ch = 'A'
for i, x in enumerate(points):
    ax.annotate(r'${}({}, {})$'.format(chr(ord(ch) + i),x[0],x[1]),(x[0]+0.02,x[1]+0.02))
# Add the patch to the Axes
'''
ax.annotate(r'$F(X,Y)=xy$', (0.3, 0.5))
ax.annotate(r'$F(X,Y)=x$', (0.3, 1.5))
ax.annotate(r'$F(X,Y)=y$', (1.25, 0.5))
ax.annotate(r'$F(X,Y)=1$', (1.25, 1.5))
'''
#plt.show()
plt.savefig('./graphs/implementation/2d/2d_rmi_limitation.pdf')