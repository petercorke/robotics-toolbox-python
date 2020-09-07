#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
# import numpy as np

# Make Robots
ur5 = rp.models.UR5()
ur5.base = sm.SE3.Ty(0.3)

puma560 = rp.models.Puma560()
puma560.base = sm.SE3.Ty(-0.3)

panda = rp.models.Panda()
panda.base = sm.SE3.Ty(0.6)

wx250s = rp.models.wx250s()
wx250s.base = sm.SE3.Ty(0.9)

# Launch Sim
env = rp.backend.Sim()
env.launch()

# Add robots
env.add(ur5)
env.add(puma560)
env.add(panda)
env.add(wx250s)
