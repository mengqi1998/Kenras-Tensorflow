
from numpy import array
from matplotlib import pyplot as plt

features = [[0, 0], [1, 1], [2, 2]]
labels = [1, 4, 8]
featuresArray = array(features)

figure = plt.figure(figsize=(8, 6))
ax = figure.add_subplot(projection='3d')
ax.scatter(featuresArray[:, 0], featuresArray[:, 1], labels, marker='*')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()