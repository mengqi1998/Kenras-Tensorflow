
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

dataset1 = np.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape, resultList.shape)
print(inputList.dtype, resultList.dtype)

model = Sequential()
model.add(Input(shape=(8,)))
model.add(Dense(14, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(inputList, resultList, validation_split=0.2, epochs=300, batch_size=20)