from matplotlib import pyplot as plt
import numpy as np

x = np.arange(0, 1, 0.0001)
y = -np.log(x)
plt.figure(0)
plt.plot(x, y, label=f"y=-log(x)")
plt.legend()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xticks([0, 0.5, 1])
plt.axis((0.0, 1.0, 0.0, 10.0))
plt.figure(1)
y = -np.log(1 - x)
plt.plot(x, y, label=f"y=-log(1-x)")
plt.xticks((0, 0.5, 1))
plt.axis((0.0, 1.0, 0.0, 10.0))
plt.legend()
plt.show()

