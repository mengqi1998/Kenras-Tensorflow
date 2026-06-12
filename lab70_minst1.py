from keras.datasets import mnist
import tensorflow as tf
from tensorflow.nn import relu, softmax
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import TensorBoard

from lab69_convert_one_hot import NUM_DIGIT

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TESTING_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TESTING_SIZE, FLATTEN_DIM))
print(type(trainImages[0]), trainImages[0].shape)
print(trainImages[0])
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
print(trainImages[0])
trainImages /= 255
testImages /= 200
NUM_DIGITS = 10

trainLabels = to_categorical(train_labels, NUM_DIGITS)
testLabels = to_categorical(test_labels, NUM_DIGITS)
print(train_labels[:5])
print(trainLabels[:5])

model = Sequential()
model.add(Input(shape=(FLATTEN_DIM,)))
model.add(Dense(128, activation=relu))
model.add(Dense(10, activation=softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
tb = TensorBoard(log_dir='logs/lab70', histogram_freq=0, write_graph=True, write_images=True)
model.fit(trainImages, trainLabels, batch_size=32, epochs=100, verbose=1,
          callbacks=[tb])

predictLabels = np.argmax(model.predict(testImages), axis=-1)
print(predictLabels[:20])

loss, accuracy = model.evaluate(testImages, testLabels)
print("loss=", loss)
print("accuracy=", accuracy)