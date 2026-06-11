
import tensorflow as tf
from datetime import datetime


@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


timestamp1 = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/lab48/%s' % timestamp1

print(logdir)

area = computeArea(tf.constant(
    [[3.0, 4.0, 5.0],
     [6.0, 6.0, 6.0],
     [3.0, 4.0, 6.0]]
))

print(area.numpy())