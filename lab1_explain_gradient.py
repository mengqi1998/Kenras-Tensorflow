from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

regressionData = datasets.make_regression(n_samples=1000, n_features=1, noise=5)
# print(regressionData)
# ^,
plt.scatter(regressionData[0], regressionData[1], c='blue', marker='*')

init_m = 10
init_b = 10
learning_rate = 0.01
range1 = [-5, 5]
plt.plot(range1, init_m * np.array(range1) + init_b, 'g--')


def cost(m, b, X, Y):
    N = len(X)
    cost = 0
    for i in range(N):
        cost += (Y[i] - (m * X[i] + b)) ** 2
    return cost / N


init_cost = cost(init_m, init_b, regressionData[0], regressionData[1])
print(init_cost, type(init_cost))
plt.title("cost={:.8f}".format(init_cost[0]))
plt.show()


def update_weight(m, b, X, Y, l):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        m_deriv += -2 * X[i] * (Y[i] - (m * X[i] + b))
        b_deriv += -2 * (Y[i] - (m * X[i] + b))
    m -= l * (m_deriv / N)
    b -= l * (b_deriv / N)
    return m, b


current_m = init_m
current_b = init_b
for i in range(201):
    new_m, new_b = update_weight(current_m, current_b, regressionData[0], regressionData[1], learning_rate)
    print(new_m, new_b)
    new_cost = cost(new_m, new_b, regressionData[0], regressionData[1])
    if i % 10 == 0:
        plt.plot(range1, new_m * range1 + new_b, 'g--')
        plt.scatter(regressionData[0], regressionData[1], c='blue', marker='*')
        plt.title("[{}]cost={:.8f}".format(i, new_cost[0]))
        plt.show()
    print("cost={}".format(new_cost))
    current_m = new_m
    current_b = new_b