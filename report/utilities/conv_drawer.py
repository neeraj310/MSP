# Copyright (c) 2021 Xiaozhe Yao et al.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

X=[(1,1),(2,2),(3,3),(5,4),(6,5),(7,6)]

plt.xlim(0,10)
plt.ylim(0,10)

for x in X:
    plt.scatter(x[0],x[1])
    plt.annotate(r'$({},{})$'.format(x[0],x[1]), (x[0]-0.2,x[1]+0.2))

x_values = [1, 3]
y_values = [1, 3]
plt.plot(x_values, y_values, color='green', linestyle='dashed')

x_values = [5, 7]
y_values = [4, 6]
plt.plot(x_values, y_values, color='blue', linestyle='dashed')

x_values = [3, 5]
y_values = [3, 4]
plt.plot(x_values, y_values, color='red', linestyle='dashed')

# plt.show()
plt.savefig('graphs/conv/points_examples.pdf', transparent=True)