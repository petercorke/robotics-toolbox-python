#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import spatialmath as sm
import numpy as np
import time

env = rp.backend.PyPlot()
env.launch('Panda Resolved-Rate Motion Control Example')

panda = rp.PandaMDH()
panda.q = panda.qr

Tep = panda.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)

arrived = False
env.add(panda)

dt = 0.05

while not arrived:

    start = time.time()
    v, arrived = rp.p_servo(panda.fkine(), Tep, 1)
    panda.qd = np.linalg.pinv(panda.jacobe()) @ v
    env.step(50)
    stop = time.time()

    if stop - start < dt:
        time.sleep(dt - (stop - start))

# Uncomment to stop the plot from closing
# env.hold()
