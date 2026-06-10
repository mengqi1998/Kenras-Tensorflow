from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from numpy import mean
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
features = iris.data
target = iris.target

classifiers = [LogisticRegression(max_iter=200),
               SVC(kernel='linear'),
               SVC(kernel='poly'),
               SVC(kernel='rbf'),
               SVC(),
               DecisionTreeClassifier()
               ]

for c in classifiers:
    scores = model_selection.cross_val_score(c, features, target, cv=3)
    print(c, scores, mean(scores))
