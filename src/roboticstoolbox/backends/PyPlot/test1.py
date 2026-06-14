import matplotlib.pyplot as plt
import spatialgeometry
from roboticstoolbox.backends.PyPlot import PyPlot
from spatialmath import SE3

env = PyPlot()
env.launch()

# box = spatialgeometry.Box(sides=[0.02,]*3)
# print(box)
# env.add(box)

# for i in range(1000):
#     box.base = SE3(i / 1000, i / 1000, i / 500) * SE3.Rx(6 * i / 500)
#     env.step()

sph = spatialgeometry.Sphere(radius=0.05)
print(sph)
env.add(sph)

for i in range(1000):
    sph.base = SE3(i / 1000, i / 1000, i / 500) * SE3.Rx(6 * i / 500)
    env.step()

# env.hold()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# from itertools import product, combinations
# from numpy import sin, cos
# from matplotlib.patches import Rectangle, Circle, PathPatch
# import mpl_toolkits.mplot3d.art3d as art3d
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from spatialmath.base import *
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect("auto")
# ax.set_autoscale_on(True)


# plot_box3d(sides=[0.2, 0.3, 0.4], wireframe=True, linewidth=1, color='r')
# # plot_box3d(sides=[0.2, 0.3, 0.4], wireframe=False, facecolors='none', linewidths=1, edgecolors='r', alpha=.25)

# plt.show(block=True)
