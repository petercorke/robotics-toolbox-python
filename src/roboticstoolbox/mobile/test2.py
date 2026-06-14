import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# a = np.zeros((10,10))
# for i in range(10):
#     a[i,:] = i / 10

# plt.imshow(a, cmap='gray')

# b = np.zeros((10,10))
# # c = np.ones((10,10))
# # c[3:6, 3:6]= 0
# b[2:7, 2:7] = 1
# b[3:6, 3:6] = 2

# colors = [(1, 1, 1, 0), (1, 0.5, 0.5, 1), (1, 0, 0, 1)]
# c_map = mpl.colors.ListedColormap(colors)

# plt.imshow(b, cmap=c_map)

# plt.show(block=True)

fig = plt.figure()
ax = fig.gca(projection="3d")

a = np.random.rand(2, 2)
X, Y = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
print(X)
print(Y)
surf = ax.plot_surface(X, Y, a, linewidth=1, antialiased=False)  # cmap='gray',
plt.show(block=True)
