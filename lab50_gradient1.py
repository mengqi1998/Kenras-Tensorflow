from tensorflow import GradientTape, Variable

x = Variable(10980.)
with GradientTape() as tape:
    y = 2 * x + 3
    dydx = tape.gradient(y, x) #微分：不論x如何變化，微分後僅會保留2
    print(dydx.numpy())