from matplotlib import pyplot as plt

figure = plt.figure(figsize=(8, 6))
ax = figure.add_subplot(projection='3d')
x = [1, 3, 5]
y = [6, 4, 2]
z = [7, 5, 6]
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()