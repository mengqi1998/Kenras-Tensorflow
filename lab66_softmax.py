import numpy as np
from tensorflow.nn import softmax

scores = [3.0, 1.0, 2.0]


def noSoftMax(x):
    ax = np.array(x)
    return ax / np.sum(ax)


print(noSoftMax(scores))


def mySoftMax(x):
    ax = np.array(x)
    aex = np.exp(ax)
    return aex / np.sum(aex)


print(mySoftMax(scores))

print(softmax(np.array(scores)).numpy())

scores2 = [8.0, 1.0, 1.0]
print(softmax(np.array(scores2)).numpy())