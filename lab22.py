from scipy.stats import alpha
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

plt.axhline(0.0, color='yellow', linestyle='-.', alpha=0.5)
plt.axhline(1.0, color='green', linestyle='-.', alpha=0.5)
plt.axhline(1.0, color='yellow', linestyle='-.')
plt.axhline(0.5, color='red', linestyle='--')
iris = datasets.load_iris()
print(iris.keys())
print(iris.feature_names)
X = iris.data
y = iris.target
X = X[:, 3:]
print(X)
y = (y == 2).astype(int)
print(y)
regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_)
print(regression1.intercept_)

X_seq = np.linspace(0, 2.6, 1000).reshape(-1, 1)
y_probability = regression1.predict_proba(X_seq)
Z = 1 / (1 + np.exp(-(regression1.coef_ * X_seq + regression1.intercept_)))
print(y_probability.shape)

plt.plot(X, y, "g.")
plt.plot(X_seq, y_probability[:, 1], "b", label="iris virginica")
plt.plot(X_seq, y_probability[:, 0], "g--", label="not iris virginica")
plt.plot(X_seq, Z, "r", label="hand calculated",alpha=0.5, linewidth=6)
plt.legend()

plt.show()
plt.show()