
from tensorflow.nn import relu, sigmoid, leaky_relu

vectors = [3.0, -1.0, 2.4, 5.9, 0.001, -0.1, 8.5, 0.000000001, -0.00000001]

r1 = relu(vectors)
print(r1)
r2 = sigmoid(vectors)
print(r2)
r3 = leaky_relu(vectors, alpha=0.1)
print(r3)