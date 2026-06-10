import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
classifier = GaussianNB()
classifier.fit(X, Y)
result = classifier.predict([[0, 0], [-1, 1], [1, -1], [0, 3], [3, 0]])
print(result)

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
result = classifier2.predict([[0, 0], [-1, 1], [1, -1], [0, 3], [3, 0]])
print("partial fit result:")
print(result)
classifier2.partial_fit([[0.5, 0.5], [0.4, 0.6], [0.6, 0.4]], [2, 2, 2])
newResult = classifier2.predict([[0, 0], [-1, 1], [1, -1], [0, 3], [3, 0]])
print(newResult)