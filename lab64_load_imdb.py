import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
Y = numpy.concatenate((y_train, y_test), axis=0)
print(X.shape, Y.shape)
print(numpy.unique(Y))
print(len(numpy.unique(numpy.hstack(X))))
result = [len(x) for x in X]
print("長度的mean={}, std={}".format(numpy.mean(result), numpy.std(result)))

plt.subplot(1, 2, 1)
plt.boxplot(result)
plt.subplot(1, 2, 2)
plt.hist(result)
plt.show()