import tensorflow as tf

tensor1 = tf.constant("hello tensorflow")
print(type(tensor1))
print(tensor1)
value1 = tensor1.numpy()
print(type(value1), value1) ##預設為class 'bytes'
x1 = "hello"
print(type(x1), x1)
x2 = b"hello"
print(type(x2), x2)
print(x1 == x2)
