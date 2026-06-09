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
# datamax = data.max(axis=0) + 0.1
# datamin = data.min(axis=0) - 0.1
datamax = data.max(axis=0) + 5
datamin = data.min(axis=0) - 5

n = 1000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
# linear, 1==>0.966666
# linear, 100==> 0.97
# poly, rbf
# sigmoid
svc = svm.SVC(kernel='linear', C=float('inf'))
svc.fit(data, iris.target)
vectors = svc.support_vectors_
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
plt.contour(X, Y, Z.reshape(X.shape))
print(vectors)
for c, s in zip([0, 1, 2], ['o', '^', 's']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
plt.scatter(vectors[:, 0], vectors[:, 1], c='purple', marker='*', alpha=0.5, s=100)
plt.title('accuracy={}'.format(svc.score(data, iris.target)))
plt.show()



# infinitive ==> float('inf')