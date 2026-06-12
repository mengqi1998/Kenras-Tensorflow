from matplotlib import pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train data shape:", train_images.shape)
print("train target shape:", train_labels.shape)
print("test data shape:", test_images.shape)
print("test target shape:", test_labels.shape)


def plotImage(index):
    plt.title("[%d]Train image: %d" % (index, train_labels[index]))
    plt.imshow(train_images[index], cmap='binary')
    plt.show()


def plotTestImage(index):
    plt.title("[%d]TestS image: %d" % (index, test_labels[index]))
    plt.imshow(test_images[index], cmap='binary')
    plt.show()

plotTestImage(9999)
plotTestImage(9998)
plotImage(1)
plotImage(10)