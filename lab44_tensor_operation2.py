import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

ta = tf.constant([5, 3, 8])
tb = tf.constant([3, -1, 2])
tc = tf.add(ta, tb)
print(tc, type(tc))

print("---start tensorflow session---")

with tf.compat.v1.Session() as session1:
    result = session1.run(tc)
    print(result)
print("---finish tensorflow session---")
