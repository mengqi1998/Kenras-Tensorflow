
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import StratifiedKFold

dataset1 = np.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)

totalScores = []


def createModel():
    m = Sequential()
    m.add(Input(shape=(8,)))
    m.add(Dense(14, activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.summary()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


for train, test in fiveFold.split(inputList, resultList):
    model = createModel()
    model.fit(inputList[train], resultList[train], epochs=300, batch_size=20, verbose=0)
    scores = model.evaluate(inputList[test], resultList[test], verbose=1)
    totalScores.append(scores[1] * 100)
    print("get a result=%.3f%%" % (scores[1] * 100))
print("average={},std={}".format(np.mean(totalScores), np.std(totalScores)))