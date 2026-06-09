
import numpy as np
from matplotlib import pyplot as plt

ws = [0.5, 1.0, 2.0, 3.0, 4.0]
message = 'w={}'

x = np.arange(-30, 30, 0.1)

for w in ws:
    y = 1 / (1 + np.exp(-w * x))
    plt.xlabel('x')
    plt.ylabel('P,F(x)')
    plt.plot(x, y,label=message.format(w))
plt.legend()
plt.show()
