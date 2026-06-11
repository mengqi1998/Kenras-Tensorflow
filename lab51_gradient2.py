from tensorflow import GradientTape, Variable
from tensorflow.random import uniform

x = Variable(uniform((2, 2)))
with GradientTape() as tape:
    y = 5 * x ** 2 + 4
    dydx = tape.gradient(y, x)
    print("x=")
    print(x.numpy())
    print("dydx=")
    print(dydx.numpy())