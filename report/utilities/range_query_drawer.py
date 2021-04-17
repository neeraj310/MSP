import matplotlib

matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

points = [(1, 2), (3, 4), (3.5, 4), (5, 6)]

plt.scatter([x[0] for x in points], [x[1] for x in points])

ax = plt.gca()

rect = Rectangle((2, 3), 3, 2, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
ax.annotate(r'$\mathcal{Q}((2,3),(5,5))$', (2, 5.1))
# plt.show()
plt.savefig('./graphs/implementation/queries/range_query.pdf')
