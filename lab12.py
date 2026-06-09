from matplotlib import pyplot as plt
from sklearn import datasets

regressionData = datasets.make_regression(10, 6, noise=5)

print(type(regressionData), regressionData[0].shape, regressionData[1].shape)

for i in range(6):
    x1 = regressionData[0][:, i]
    y = regressionData[1]

    plt.scatter(x1, y)
    plt.show()