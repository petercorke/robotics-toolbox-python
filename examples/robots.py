#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

# Make Robots
panda = rp.models.Panda()
panda.base = sm.SE3.Ty(0)
panda.q = panda.qr

ur3 = rp.models.UR3()
ur3.base = sm.SE3.Tx(0) * sm.SE3.Ty(0.4)
ur3.q = [0, -np.pi / 2, np.pi / 2, 0, 0, 0]

ur5 = rp.models.UR5()
ur5.base = sm.SE3.Tx(0) * sm.SE3.Ty(0.8)
ur5.q = [0, -np.pi / 2, np.pi / 2, 0, 0, 0]

ur10 = rp.models.UR10()
ur10.base = sm.SE3.Tx(0) * sm.SE3.Ty(1.2)
ur10.q = [0, -np.pi / 2, np.pi / 2, 0, 0, 0]

puma560 = rp.models.Puma560()
puma560.base = sm.SE3.Ty(2.0)

# Interbotix robots
px100 = rp.models.px100()
px100.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(0)
px150 = rp.models.px150()
px150.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(0.3)
rx150 = rp.models.rx150()
rx150.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(0.6)
rx200 = rp.models.rx200()
rx200.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(0.9)
vx300 = rp.models.vx300()
vx300.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(1.2)
vx300s = rp.models.vx300s()
vx300s.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(1.5)
wx200 = rp.models.wx200()
wx200.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(1.8)
wx250 = rp.models.wx250()
wx250.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(2.1)
wx250s = rp.models.wx250s()
wx250s.base = sm.SE3.Tx(0.7) * sm.SE3.Ty(2.4)

# # Kinova robots
# j2n4s300 = rp.models.j2n4s300()
# j2n4s300.base = sm.SE3.Tx(1.2) * sm.SE3.Ty(0)
# j2n4s300.q = j2n4s300.qr

# Launch Sim
env = swift.Swift()
env.launch()

# Add robots
env.add(ur3)
env.add(ur5)
env.add(ur10)
env.add(puma560)
env.add(panda)

env.add(px100)
env.add(px150)
env.add(rx150)
env.add(rx200)
env.add(vx300)
env.add(vx300s)
env.add(wx200)
env.add(wx250)
env.add(wx250s)

# env.add(j2n4s300)
env.hold()
