#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import time

# Launch the simulator Swift
env = rtb.backends.Swift()
env.launch()

# Create a Panda robot object
r = rtb.models.Frankie()

# Set joint angles to ready configuration
r.q = r.qr

# Add the puma to the simulator
env.add(r)
time.sleep(1)

Tep = r.fkine(r.q) * sm.SE3.Tx(1.0) * sm.SE3.Ty(1.0)

arrived = False

dt = 0.05

while not arrived:
    v, arrived = rtb.p_servo(r.fkine(r.q), Tep, 0.1)
    r.qd = np.linalg.pinv(r.jacobe(r.q)) @ v
    env.step(dt)
