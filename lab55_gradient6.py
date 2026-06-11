import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

NUM_SAMPLES_PER_CLASS = 1000

negativeSamples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=NUM_SAMPLES_PER_CLASS
)#如果mean改成[0,5],且positiveSamples的mean改成[-5,0] 圖形就會較分開，較無重疊的點

positiveSamples = np.random.multivariate_normal(
    mean=[-3, 0], cov=[[1, 0.5], [0.5, 1]], size=NUM_SAMPLES_PER_CLASS
)

inputs = np.vstack((negativeSamples, positiveSamples)).astype(np.float32)

negativeTargets = np.zeros((NUM_SAMPLES_PER_CLASS, 1), dtype=np.float32)
positiveTargets = np.ones((NUM_SAMPLES_PER_CLASS, 1), dtype=np.float32)
targets = np.vstack((negativeTargets, positiveTargets))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])

inputs = np.vstack((negativeSamples, positiveSamples)).astype(np.float32)

negativeTargets = np.zeros((NUM_SAMPLES_PER_CLASS, 1), dtype=np.float32)
positiveTargets = np.ones((NUM_SAMPLES_PER_CLASS, 1), dtype=np.float32)
targets = np.vstack((negativeTargets, positiveTargets))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.title("generated real data")
input_dim = 2
output_dim = 1
W = tf.Variable(np.random.uniform(size=(input_dim, output_dim)), dtype=tf.float32)
b = tf.Variable(tf.zeros(shape=(output_dim,)))


def model(inputs):
    return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


LEARNING_RATE = 0.01


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
        d_loss_d_w, d_loss_d_b = tape.gradient(loss, [W, b])
        W.assign_sub(d_loss_d_w * LEARNING_RATE)
        b.assign_sub(d_loss_d_b * LEARNING_RATE)
        return loss


EPOCHS = 300
for step in range(EPOCHS):
    loss = training_step(inputs, targets)
    print(f"loss at step:{step} value:{loss:.4f}")

predictions = model(inputs)
x = np.linspace(-7, 3, 100)
plt.figure(2)
# W[0]x + W[1]y+b = y
# W[0]x + W[1]y+b = 0.5
y1 = (0.5 - b) / W[1] - W[0] / W[1] * x
y2 = (0 - b) / W[1] - W[0] / W[1] * x
y3 = (1 - b) / W[1] - W[0] / W[1] * x
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'g--')
plt.plot(x, y3, 'g--')
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.title("mse predict result")
plt.show()