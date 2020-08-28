#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import spatialmath as sm
import numpy as np
import time

env = rp.backend.Sim()
env.launch()

wx = rp.wx250s()
# wx.q = [0, 0, 0, 0, 0, 0]

# Tep = wx.fkine() * sm.SE3.Tx(-0.05) * sm.SE3.Ty(0.05) * sm.SE3.Tz(0.05)

# arrived = False
# env.add(wx)

# dt = 50

# for link in wx.ets:
#     print()
#     print(link.name)
#     # print(link._fk)
#     for g in link.geometry:
#         print(g.base)

# while not arrived:

#     start = time.time()
#     v, arrived = rp.p_servo(wx.fkine(), Tep, 1)
#     wx.qd = np.linalg.pinv(wx.jacobe()) @ v
#     env.step(dt)
#     stop = time.time()

#     if stop - start < dt:
#         time.sleep(dt - (stop - start))

# Uncomment to stop the plot from closing
# env.hold()
