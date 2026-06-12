import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scikeras.wrappers import KerasClassifier

dataset1 = np.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]


def createModel():
    m = Sequential()
    m.add(Input(shape=(8,)))
    m.add(Dense(14, activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.summary()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


model = KerasClassifier(model=createModel, epochs=300, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
result = cross_val_score(model, inputList, resultList, cv=fiveFold)
print("mean=%.3f, std=%.3f"%(result.mean(), result.std()))