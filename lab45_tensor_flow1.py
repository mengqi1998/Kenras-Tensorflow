
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
b = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
print(type(a))
print(type(b))
print(type(c))

with tf.compat.v1.Session() as sess:
    result = sess.run(c, feed_dict={
        # a: [1, 2, 3, 4, 5],
        # b: [9, 7, 5, 3, 1]
        # a: [3, 11],
        # b: [5, 1]
        a: [1.2, 2.4, 3.9],
        b: [4.4, 5.5, 6.9]
    })
    print(result)