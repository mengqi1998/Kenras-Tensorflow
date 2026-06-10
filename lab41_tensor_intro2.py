
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tensor1 = tf.constant('hello tensorflow')
print(type(tensor1))
print(tensor1)
s1 = tf.compat.v1.Session()
result = s1.run(tensor1)
print(result)
s1.close()