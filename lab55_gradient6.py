
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

NUM_SAMPLES_PER_CLASS = 1000

negativeSamples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=NUM_SAMPLES_PER_CLASS
) #如果mean改成[0,5],且positiveSamples的mean改成[-5,0] 圖形就會較分開，較無重疊的點

positiveSamples = np.random.multivariate_normal(
    mean=[-3, 0], cov=[[1, 0.5], [0.5, 1]], size=NUM_SAMPLES_PER_CLASS
)

inputs = np.vstack((negativeSamples, positiveSamples)).astype(np.float32)

negativeTargets = np.zeros((NUM_SAMPLES_PER_CLASS, 1), dtype=np.float32)
positiveTargets = np.ones((NUM_SAMPLES_PER_CLASS, 1), dtype=np.float32)
targets = np.vstack((negativeTargets, positiveTargets))

plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()