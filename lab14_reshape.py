import numpy as np

a = np.zeros((10, 2))
print(a.shape)
print(a)
b = a.T
c = b.view()
print("b shape={}, c shape={}".format(b.shape, c.shape))
d = np.reshape(b, (5, 4))
print("b shape={}, d shape={}, c shape={}".format(b.shape, d.shape, c.shape))
e = np.reshape(b, (20,))
print(e.shape, e)
f = np.reshape(b, (20, -1))
print(f.shape)
print(f)
g = np.reshape(b, (-1, 20))
print(g.shape)
print(g)