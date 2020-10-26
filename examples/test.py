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
env = rtb.backend.Swift()
env.launch()

# Create a Panda robot object
puma = rtb.models.Puma560()

# Set joint angles to ready configuration
puma.q = puma.qr

# Add the puma to the simulator
env.add(puma)

# Tep = puma.fkine() * sm.SE3.Tz(0.1)

# arrived = False

# dt = 0.05

# while not arrived:

#     start = time.time()
#     v, arrived = rtb.p_servo(puma.fkine(), Tep, 0.1)
#     puma.qd = np.linalg.pinv(puma.jacobe()) @ v
#     env.step(50)
#     stop = time.time()

#     if stop - start < dt:
#         time.sleep(dt - (stop - start))
