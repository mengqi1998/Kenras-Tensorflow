
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from numpy import mean
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
features = iris.data
target = iris.target

classifiers = [LogisticRegression(max_iter=200),
               SVC(kernel='linear'),
               SVC(kernel='poly'),
               SVC(kernel='rbf'),
               SVC(),
               DecisionTreeClassifier(),
               KNeighborsClassifier(n_neighbors=2),
               KNeighborsClassifier(n_neighbors=3),
               KNeighborsClassifier(n_neighbors=4),
               KNeighborsClassifier(n_neighbors=5),
               KNeighborsClassifier(n_neighbors=6),
               KNeighborsClassifier(n_neighbors=7),
               KNeighborsClassifier(n_neighbors=8),
               KNeighborsClassifier(n_neighbors=9),
               ]

for c in classifiers:
    scores = model_selection.cross_val_score(c, features, target, cv=3)
    print(c, scores, mean(scores))