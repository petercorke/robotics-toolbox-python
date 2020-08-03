
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


# env = rp.PyPlot()
# env.launch()

puma = rp.Puma560()
puma.q = puma.qn

# env.add(puma)

# env.hold()

# puma.jacob_dot(puma.qr, puma.qr)

a = puma.maniplty()

print(a)
# print(b)
