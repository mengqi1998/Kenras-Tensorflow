
from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(dir(iris))
features = iris.feature_names
print(features)
X = iris.data
y = iris.target

counter = 1
for i in range(0, 4):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(6, 6))
        counter += 1
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - 0.1, xData.max() + 0.1
        y_min, y_max = yData.min() - 0.1, yData.max() + 0.1
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.scatter(xData, yData, c=y)
plt.show()