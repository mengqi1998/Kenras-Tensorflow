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
#
# model.fit(inputList, resultList, epochs=300, batch_size=20, verbose=0)
# scores = model.evaluate(inputList, resultList)
# print("訓練過的model")
# print(scores)
# # 手動建立models的目錄
# save_model(model, 'models/lab58.keras')
#
# model2 = createModel()
# scores2 = model2.evaluate(inputList, resultList)
# print("初始化未訓練的模型")
# print(scores2)
#
# model3 = load_model('models/lab58.keras')
# scores3 = model3.evaluate(inputList, resultList)
# print("未訓練直接載入訓練好的模型")
# print(scores3)
#
# plot_model(model3, 'output/lab58_model3.png', show_shapes=True, show_layer_names=True)