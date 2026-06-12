import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.nn import relu, softmax

csv = pd.read_csv('data/bmi.csv')
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv['label'])
print(csv['label'][:10])
print(transformedLabel[:10])

test_csv = csv[25000:]
test_ans = transformedLabel[25000:]

train_csv = csv[:25000]
train_ans = transformedLabel[:25000]

test_pat = test_csv[['weight', 'height']]
train_pat = train_csv[['weight', 'height']]

model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(10, activation=relu))
model.add(Dense(3, activation=softmax))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

tb = TensorBoard(log_dir="logs/lab72", histogram_freq=1, write_graph=False)
history = model.fit(train_pat, train_ans, batch_size=32, epochs=200, validation_data=(test_pat, test_ans),
                    callbacks=[tb])

score = model.evaluate(test_pat, test_ans, verbose=1)
print(score)