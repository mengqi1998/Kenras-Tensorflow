
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(-30, 30, 0.1)
y = 1 / (1 + np.exp(-x))
plt.xlabel('x')

plt.ylabel('P,F(x)')
plt.plot(x, y)
plt.axhline(0.0, color='green', linestyle='-.')
plt.axhline(1.0, color='green', linestyle='-.')
plt.axhline(0.5, color='red', linestyle='--')
plt.axvline(0, color='black')
plt.show()