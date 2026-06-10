from numpy import array
from sklearn.decomposition import PCA

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
pca = PCA(n_components=2)
pca.fit(A)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.components_)
B = pca.transform(A)
print(B)