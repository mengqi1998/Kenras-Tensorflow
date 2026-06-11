
from tensorflow import GradientTape, Variable, zeros, matmul, constant, square
from tensorflow.random import uniform

x = constant(3.)
# x = Variable(3.)
with GradientTape() as tape:
    tape.watch(x)
    y = square(x)
    dydx = tape.gradient(y, x)
    print(dydx.numpy())