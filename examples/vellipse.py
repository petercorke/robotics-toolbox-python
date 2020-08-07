#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import numpy as np
import spatialmath as sm
from ropy.connect.PyPlot import Ellipse
import time

env = rp.PyPlot()
env.launch('Panda Resolved-Rate Motion Control Example')

panda = rp.PandaMDH()
panda.q = panda.qr

Tep = panda.fkine() * sm.SE3.Tx(-0.4) * sm.SE3.Tz(0.4) * sm.SE3.Rz(np.pi) 

arrived = False
env.add(panda)

ell = Ellipse(env.ax)

dt = 0.05

while not arrived:
    start = time.time()

    A = panda.jacobe()[3:, 3:] @ np.transpose(panda.jacobe()[3:, 3:])
    ell.make_ellipsoid(A, centre=panda.fkine().t)
    ell.draw_ellipsoid()

    v, arrived = rp.p_servo(panda.fkine(), Tep, 0.5)
    panda.qd = np.linalg.pinv(panda.jacobe()) @ v
    env.step(50)
    stop = time.time()

    if stop - start < dt:
        print("Hello")
        time.sleep(dt - (stop - start))

env.hold()
