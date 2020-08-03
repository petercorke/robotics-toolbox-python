
import ropy as rp
# import spatialmath as sm
# import numpy as np
# import ctypes
# import frne
# import matplotlib.pyplot as plt

# import time

# T1 = sm.SE3.Tx(1)
# plt.figure() # create a new figure
# sm.SE3().plot(frame='0', dims=[-3,3], color='black')
# T1.plot(frame='1')


# env1 = rp.PyPlot()
# env2 = rp.ROS()
# env.launch()

puma = rp.PandaMDH()
puma.q = puma.qr

# puma.q = [0, -1.57079632679490, -1.57079632679490, 1.57079632679490, 0, -1.57079632679490, 1.57079632679490]

puma.qmincon()

# env1.add(puma, 'vis')
# env2.add(puma)

# while q:
#     delay(x)
#     env.Step()

# env.hold()

# puma.q

# puma.jacob_dot(puma.qr, puma.qr)

# a = puma.maniplty()
