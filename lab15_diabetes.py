import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

data = datasets.load_diabetes()
print(type(data))
print(dir(data))
print(data.data.shape)
print(data.target.shape)
print(data.target)

dataForTest = -60
data_train = data.data[:dataForTest]
target_train = data.target[:dataForTest]
print("train X shape:{}, y shape:{}".format(data_train.shape, target_train.shape))

data_test = data.data[dataForTest:]
target_test = data.target[dataForTest:]
print("test X shape:{}, y shape:{}".format(data_test.shape, target_test.shape))

regression1 = LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)
print(regression1.score(data_test, target_test))

# evaluate test data
for i in range(dataForTest, 0):
    d = np.array(data_test[i]).reshape(1, -1)
    p = regression1.predict(d)
    print("predict:{:.2f}, actual:{:.2f}".format(p[0], target_test[i]))
mse = np.mean((regression1.predict(data_test) - target_test) ** 2)
print("mse:{:.2f}".format(mse))