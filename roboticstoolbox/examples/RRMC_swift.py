#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch()

panda = rp.models.Panda()
panda.q = panda.qr

Tep = panda.fkine(panda.q) * sm.SE3.Tx(0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.45)

arrived = False
env.add(panda)

dt = 0.05

while not arrived:

    v, arrived = rp.p_servo(panda.fkine(panda.q), Tep, 1)
    panda.qd = np.linalg.pinv(panda.jacobe(panda.q)) @ v
    env.step(dt)

# Uncomment to stop the browser tab from closing
# env.hold()
