
import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

nn = NearestNeighbors(n_neighbors=3).fit(X)
print(nn)
distances, indices = nn.kneighbors(X)
print(distances)
print(indices)