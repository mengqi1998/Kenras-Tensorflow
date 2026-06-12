from keras import Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.nn import relu, softmax

model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation=relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation=relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation=relu))
model.add(Flatten())
model.add(Dense(64, activation=relu))
model.add(Dense(10, activation=softmax))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')
train_images /= 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32')
test_images /= 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.fit(train_images, train_labels, batch_size=32, epochs=10)
score = model.evaluate(test_images, test_labels)
print(score)
