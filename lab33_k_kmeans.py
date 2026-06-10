

import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


CLUSTER_COUNT = 50000
X = np.r_[np.random.randn(CLUSTER_COUNT, 2) + [2, 2],
          np.random.randn(CLUSTER_COUNT, 2) + [0, -2],
          np.random.randn(CLUSTER_COUNT, 2) + [-2, 2]]

inertias = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

print(inertias)
plt.plot(range(1, 11), inertias)
plt.xticks(range(1, 11))
plt.show()