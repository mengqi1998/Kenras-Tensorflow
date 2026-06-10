
import tensorflow as tf
import numpy as np

a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print(c, type(c))
d = tf.add(a, b)
print(d.numpy(), type(d))

ta = tf.constant([5, 3, 8])
tb = tf.constant([3, -1, 2])
tc = tf.add(ta, tb)
print(tc.numpy(), type(tc))
td = tf.math.add(ta, tb)
print(td.numpy(), type(td))
te = np.add(ta, tb)
print(te, type(te))
