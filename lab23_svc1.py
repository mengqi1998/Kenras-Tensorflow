
import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -3], [1, 1], [2, 1], [3, 3]])
y = np.array([1, 1, 1, 2, 2, 2])
classifier = SVC(kernel='linear')
classifier.fit(X, y)
print(classifier.coef_)
print(classifier.intercept_)
print(classifier.support_vectors_)
print(classifier.predict([[0.5, -0.5], [-0.5, 0.5], [0.3, 3], [3, 0.3], [-0.3, -3]]))