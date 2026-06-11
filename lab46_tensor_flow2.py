
import tensorflow as tf


@tf.function
def add(a, b):
    return tf.math.add(a, b)


print(add(3, 5))
print(add([1, 2, 3, 4, 5], [9, 7, 5, 3, 1]))
print(add([3, 11], [5, 1]))
print(add([1.2, 2.4, 3.9], [4.4, 5.5, 6.9]))