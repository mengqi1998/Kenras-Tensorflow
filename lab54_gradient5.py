from tensorflow import Variable, GradientTape

time = Variable(5.)
with GradientTape() as outerTape:
    with GradientTape() as innerTape:
        position = 4.9 * time ** 2
        speed = innerTape.gradient(position, time)
        print("speed=", speed.numpy())
    a = outerTape.gradient(speed, time)
    print("a=", a.numpy())