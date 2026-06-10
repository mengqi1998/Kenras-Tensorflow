from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

X = iris.data
species = iris.target

fig = plt.figure(1, figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
pca3 = PCA(n_components=3)
X_reduced = pca3.fit_transform(X)
print(pca3.explained_variance_ratio_)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=species,
           cmap=plt.cm.Paired)
ax.set_title("iris in 3 dimensions")
ax.set_xlabel("first eigenvector")
ax.set_ylabel("second eigenvector")
ax.set_zlabel("third eigenvector")
plt.show()
