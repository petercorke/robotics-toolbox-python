#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

# Make Robots
ur3 = rp.models.UR3()
ur3.base = sm.SE3.Tx(0) * sm.SE3.Ty(0)
ur3.q = [0, -np.pi/2, np.pi/2, 0, 0, 0]

ur5 = rp.models.UR5()
ur5.base = sm.SE3.Tx(0) * sm.SE3.Ty(0.4)
ur5.q = [0, -np.pi/2, np.pi/2, 0, 0, 0]

ur10 = rp.models.UR10()
ur10.base = sm.SE3.Tx(0) * sm.SE3.Ty(0.8)
ur10.q = [0, -np.pi/2, np.pi/2, 0, 0, 0]

puma560 = rp.models.Puma560()
puma560.base = sm.SE3.Ty(-0.4)

panda = rp.models.Panda()
panda.base = sm.SE3.Ty(1.4)
panda.q = panda.qr

wx250s = rp.models.wx250s()
wx250s.base = sm.SE3.Ty(-1)

# Launch Sim
env = rp.backend.Sim()
env.launch()

# Add robots
env.add(ur3)
env.add(ur5)
env.add(ur10)
env.add(puma560)
env.add(panda)
env.add(wx250s)
