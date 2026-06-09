import numpy as np
from matplotlib import pyplot as plt

bs = [-12.0, -8.0, -4.0, 0, 4.0, 8.0, 12.0]
message = 'b={}'

x = np.arange(-10, 10, 0.1)

for b in bs:
    y = 1 / (1 + np.exp(-(2 * x + b)))
    plt.xlabel('x')
    plt.ylabel('P,F(x)')
    plt.plot(x, y, label=message.format(b))
plt.axhline(0.0, color='green', linestyle='-.')
plt.axhline(1.0, color='green', linestyle='-.')
plt.axhline(0.5, color='red', linestyle='--')
plt.axvline(0, color='black')
plt.legend()
plt.show()

