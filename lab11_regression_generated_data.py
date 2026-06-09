
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import numpy as np

# np.random.seed(20260609) # only when you need to reproduce same result
data = datasets.make_regression(100, 1, noise=1)
print(type(data), len(data), type(data[0]), type(data[1]))
print(data[0].shape, data[1].shape)
plt.scatter(data[0], data[1], marker='.', c='blue')
regression1 = LinearRegression()
regression1.fit(data[0], data[1])
print(regression1.coef_, regression1.intercept_)
range1 = np.arange(data[0].min() - 0.5, data[0].max() + 0.5, 0.01)
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, color='red')
plt.title("score={}".format(regression1.score(data[0], data[1])))
plt.show()