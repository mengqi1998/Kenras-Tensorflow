from sklearn import datasets
from matplotlib import pyplot as plt

regressionData = datasets.make_regression(n_samples=1000,n_features=1, noise=5)
#print(regressionData)
# marker ^,.
plt.scatter(regressionData[0],regressionData[1],c='green',marker='*')
plt.show()