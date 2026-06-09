import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

regressionData = datasets.make_regression(10, 6, noise=5)

# print(type(regressionData), regressionData[0].shape, regressionData[1].shape)

l1 = [5, 1, 3, 2, 4]
sortedL1 = sorted(l1)
print(l1, sortedL1)

print(regressionData[0])
regX = regressionData[0]
result1 = sorted(regX, key=lambda t: t[0])
result1 = np.array(result1)
print(result1)

result2 = sorted(regX, key=lambda t: t[1])
result2 = np.array(result2)
print(result2)
