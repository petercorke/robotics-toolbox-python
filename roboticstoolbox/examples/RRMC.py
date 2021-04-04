#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot import PyPlot
import spatialmath as sm
import numpy as np
import time

env = PyPlot()
env.launch('Panda Resolved-Rate Motion Control Example')

panda = rtb.models.DH.Panda()
panda.q = panda.qr

Tep = panda.fkine(panda.q) * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.2)

arrived = False
env.add(panda)

dt = 0.05

while not arrived:

    start = time.time()
    v, arrived = rtb.p_servo(panda.fkine(panda.q), Tep, 1)
    panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
    env.step(dt)
    stop = time.time()

    if stop - start < dt:
        time.sleep(dt - (stop - start))

# Uncomment to stop the plot from closing
# env.hold()
