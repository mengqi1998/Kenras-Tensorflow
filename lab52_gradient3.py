from tensorflow import GradientTape, Variable, zeros, matmul
from tensorflow.random import uniform

W = Variable(uniform((1, 1)))
b = Variable(zeros((1,)))
x = uniform((1, 1))
with GradientTape() as tape:
    y = matmul(x, W) + 2 * b
    dydwdb = tape.gradient(y, [W, b])
print(W.numpy())
print("x=", x.numpy())
print(b.numpy())
print(y.numpy())
print("partial y/partial w", dydwdb[0].numpy())
print("partial y/partial b", dydwdb[1].numpy())