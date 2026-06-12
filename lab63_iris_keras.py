from pandas import read_csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

dataFrame = read_csv('data/iris.data', header=None)
print(dataFrame.shape)
dataset = dataFrame.values
print(type(dataset))
print(dataset.shape)
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features.shape)
print(labels.shape)
print(np.unique(labels))

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(np.unique(encoded_Y))
dummy_y = to_categorical(encoded_Y)
print(dummy_y[:10])

def baseline_model():
    m = Sequential()
    m.add(Input(shape=(4,)))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(3, activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m
model = baseline_model()

estimator = KerasClassifier(model=baseline_model(),
                            epochs=800, batch_size=10, verbose=0)
Kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=Kfold)
print("Acc: %.4f%%, std:%.4f"%(results.mean()*100, results.std()))