#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time
import qpsolvers as qp
import fcl

s1 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 0.20))

s2 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 0.60))

s3 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 1.00))

s4 = rp.Shape.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(0.95, 0, 1.40))

s5 = rp.Shape.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(0.95, 0.55, 0.7))

s6 = rp.Shape.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(0.95, -0.55, 0.7))

env = rp.backend.Swift()
env.launch()

# panda = rp.models.Panda()
# panda.q = panda.qr

r = rp.models.PR2()

env.add(r)
env.add(s1)
env.add(s2)
env.add(s3)
env.add(s4)
env.add(s5)
env.add(s6)
time.sleep(1)



