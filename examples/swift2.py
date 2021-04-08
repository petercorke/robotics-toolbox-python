#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

env = swift.Swift()
env.launch()


# ur = rtb.models.UR5()
# ur.base = sm.SE3(0.3, 1, 0) * sm.SE3.Rz(np.pi / 2)
# ur.q = [-0.4, -np.pi / 2 - 0.3, np.pi / 2 + 0.3, -np.pi / 2, -np.pi / 2, 0]
# env.add(ur)

# lbr = rtb.models.LBR()
# lbr.base = sm.SE3(1.8, 1, 0) * sm.SE3.Rz(np.pi / 2)
# lbr.q = lbr.qr
# env.add(lbr)

# k = rtb.models.KinovaGen3()
# k.q = k.qr
# k.q[0] = np.pi + 0.15
# k.base = sm.SE3(0.7, 1, 0) * sm.SE3.Rz(np.pi / 2)
# env.add(k)

# panda = rtb.models.Panda()
# panda.q = panda.qr
# panda.base = sm.SE3(1.2, 1, 0) * sm.SE3.Rz(np.pi / 2)
# env.add(panda, show_robot=True)

r = rtb.models.YuMi()
env.add(r)


env.hold()
