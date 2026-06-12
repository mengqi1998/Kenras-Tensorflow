
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
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