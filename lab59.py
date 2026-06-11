import numpy as np
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Input
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

dataset1 = np.loadtxt('data/diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape, resultList.shape)
print(inputList.dtype, resultList.dtype)

# split data
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList, test_size=0.2,
                                                                        stratify=resultList)
print(feature_train.shape, feature_test.shape, label_train.shape, label_test.shape)

# validate ratio
for data in [resultList, label_train, label_test]:
    classes, counts = np.unique(data, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"類別:{int(cl)}, 個數是:{co / sum(counts):.3f}")


def createModel():
    m = Sequential()
    m.add(Input(shape=(8,)))
    m.add(Dense(14, activation='relu'))
    m.add(Dense(10, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.summary()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


model = createModel()

model.fit(feature_train, label_train, epochs=300, batch_size=20,
          validation_data=(feature_test, label_test),
          verbose=1)
scores = model.evaluate(feature_test, label_test)
print("訓練過的model")
print(model.metrics_names)
print(scores)