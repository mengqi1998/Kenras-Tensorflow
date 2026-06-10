
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

k = 3
#np.random.seed(20260610)
X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

print("centers: ", kmeans.cluster_centers_)
print("inertia", kmeans.inertia_)

colors = ['c', 'm', 'y', 'k']
markers = ['o', '^', '*', '.']
for i in range(k):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
    print("[{}] size={}".format(i, dataX.size))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, marker="*", c="#0599FF")
plt.show()