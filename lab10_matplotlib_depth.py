from numpy import array
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

features = [[0, 0], [1, 1], [2, 2]]
labels = [1, 4, 8]
regression1 = LinearRegression()
regression1.fit(features, labels)
print("coef_:", regression1.coef_)
print("intercept_:", regression1.intercept_)
featuresArray = array(features)

X = np.arange(-1, 3, 0.1)
Y = np.arange(-1, 3, 0.1)
X, Y = np.meshgrid(X, Y)
Z = regression1.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)
print(X.shape, Y.shape, Z.shape)

figure = plt.figure(figsize=(8, 6))
ax = figure.add_subplot(projection='3d')
# viridis, plasma, inferno,
# https://matplotlib.org/stable/users/explain/colors/colormaps.html

ax.plot_surface(X, Y, Z,cmap=plt.cm.magma)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

