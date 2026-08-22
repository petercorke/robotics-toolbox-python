#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch(realtime=True)

panda = rp.models.Panda()
handle = env.add(panda)
handle.q = panda.qr

Tep = panda.fkine(handle.q) * sm.SE3.Tx(0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.45)

arrived = False

dt = 0.05

while not arrived:
    v, arrived = rp.p_servo(panda.fkine(handle.q), Tep, 1)
    handle.qd = np.linalg.pinv(panda.jacobe(handle.q)) @ v
    env.step(dt)

# Uncomment to stop the browser tab from closing
# env.hold()
