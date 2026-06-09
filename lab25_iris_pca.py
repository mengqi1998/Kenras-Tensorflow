
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(data.shape)
# print(data[0:5,])
datamax = data.max(axis=0) + 0.1
datamin = data.min(axis=0) - 0.1

n = 200
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
svc = svm.SVC(kernel='linear', C=1)
svc.fit(data, iris.target)
vectors = svc.support_vectors_
print(vectors)
plt.scatter(vectors[:, 0], vectors[:, 1], c='PURPLE', marker='*')
plt.show()