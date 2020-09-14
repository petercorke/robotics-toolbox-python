#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time

env = rp.backend.Swift()
env.launch()

panda = rp.models.Panda()
panda.q = panda.qr

Tep = panda.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)

arrived = False
env.add(panda)
time.sleep(1)

dt = 0.05

# env.record_start('file.webm')

# while not arrived:

#     start = time.time()
#     v, arrived = rp.p_servo(panda.fkine(), Tep, 1.0)
#     panda.qd = np.linalg.pinv(panda.jacobe()) @ v
#     env.step(5)
#     stop = time.time()

#     if stop - start < dt:
#         time.sleep(dt - (stop - start))

# env.record_stop()

# Uncomment to stop the plot from closing
# env.hold()
