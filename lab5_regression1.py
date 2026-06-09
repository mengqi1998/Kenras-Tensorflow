from sklearn import linear_model
from matplotlib import pyplot as plt

regression = linear_model.LinearRegression()
features = [[2], [5], [9]]
values = [1, 4, 5]
plt.scatter(features, values, color='blue')
regression.fit(features, values)
print(regression.coef_, regression.intercept_)
range1 = [0,10]
plt.plot(range1, regression.coef_*range1+regression.intercept_, color='red')
plt.title('Linear Regression score={}'.format(regression.score(features, values)))
plt.show()