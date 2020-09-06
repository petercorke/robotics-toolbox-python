#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import numpy as np
import spatialmath as sm
import time

panda = rp.PandaMDH()
panda.q = panda.qr

vell = panda.vellipse(centre='ee')

env = rp.backend.PyPlot()
env.launch(
    'Panda Velocity Ellipse Example',
    limits=[-0.2, 0.6, -0.4, 0.4, 0, 0.8])

Tep = panda.fkine() * sm.SE3.Tx(-0.3) * sm.SE3.Tz(0.3) * sm.SE3.Rz(np.pi)

arrived = False

env.add(panda)
env.add(vell)

dt = 0.05

while not arrived:
    start = time.time()

    v, arrived = rp.p_servo(panda.fkine(), Tep, 0.5)
    panda.qd = np.linalg.pinv(panda.jacobe()) @ v
    env.step(50)
    stop = time.time()

    if stop - start < dt:
        time.sleep(dt - (stop - start))

env.hold()
