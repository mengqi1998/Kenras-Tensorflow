from sklearn.linear_model import LinearRegression

features = [[0, 0], [1, 1], [2, 2]]
labels = [1, 4, 8]

regression1 = LinearRegression()
regression1.fit(features, labels)
print("coef_:", regression1.coef_)
print("intercept_:", regression1.intercept_)

newFeatures = [[0, 1], [1, 0], [2, 4], [-3, 5]]
predictResult = regression1.predict(newFeatures)
print(predictResult)
print("score:", regrelab6_regression2.pyssion1.score(features, labels))