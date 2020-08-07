#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import numpy as np
import spatialmath as sm
# from ropy.backend.PyPlot.PyPlot import Ellipse
import time

panda = rp.PandaMDH()
panda.q = panda.qr
# panda.q = [0, -3, 0, -2.3, 0, 2, 0]

vell = panda.vellipse(centre='ee')

env = rp.backend.PyPlot()
env.launch('Panda Velocity Ellipse Example')

Tep = panda.fkine() * sm.SE3.Tx(-0.4) * sm.SE3.Tz(0.4) * sm.SE3.Rz(np.pi)

arrived = False

env.add(panda)
env.add(vell)

dt = 0.05

while not arrived:
    start = time.time()

    # A = panda.jacobe()[3:, 3:] @ np.transpose(panda.jacobe()[3:, 3:])
    # ell.make_ellipsoid(A, centre=panda.fkine().t)
    # ell.draw_ellipsoid(env.ax)

    v, arrived = rp.p_servo(panda.fkine(), Tep, 0.5)
    panda.qd = np.linalg.pinv(panda.jacobe()) @ v
    env.step(50)
    stop = time.time()

    if stop - start < dt:
        time.sleep(dt - (stop - start))

env.hold()
