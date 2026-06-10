
from sklearn import tree
from matplotlib import pyplot as plt
X = [[0, 0], [1, 1]]
Y = [0, 1]
classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
print(classifier)
print(classifier.tree_)
print(classifier.predict([[0, 5], [0, -5], [-5, 0], [-5, -5], [5, 5], [3, 5]]))
tree.plot_tree(classifier, filled=True)
plt.show()
# https://graphviz.org/download/
# https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/15.0.0/windows_10_cmake_Release_graphviz-install-15.0.0-win64.exe