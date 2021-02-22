'''
This script shows how a monotonically increasing dataset leads to non monotonical piecewise linear function.
More strictly speaking, the piecewise linear function will be monotonic on each piece, but not the whole domain.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([])
Y = np.array([])

for x in range(10):
    X = np.append(X, x)
    Y = np.append(Y, np.power(x, 3))

breakpoint = 3

xs_1 = X[0:breakpoint + 1]
ys_1 = Y[0:breakpoint + 1]
xs_2 = X[breakpoint:]
ys_2 = Y[breakpoint:]
lr1 = LinearRegression().fit(xs_1.reshape(-1, 1), ys_1)
lr2 = LinearRegression().fit(xs_2.reshape(-1, 1), ys_2)

print(lr1.coef_)
print(lr1.intercept_)

plt.plot(X, Y)
plt.plot(xs_1, lr1.coef_ * xs_1 + lr1.intercept_, 'r')
plt.plot(xs_2, lr2.coef_ * xs_2 + lr2.intercept_, 'g')
plt.show()
