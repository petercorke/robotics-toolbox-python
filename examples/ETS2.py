#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import spatialmath as sm
import numpy as np
import time
import qpsolvers as qp

env = rp.backend.PyPlot()
env.launch('Panda Resolved-Rate Motion Control Example', limits=[-0.75, 0.75, -0.75, 0.75, 0, 1.5])

gPanda = rp.Panda()
gPanda.name = "Null"
gPanda.q = gPanda.qr

qPanda = rp.Panda()
qPanda.name = "Quad"
qPanda.q = gPanda.q

eTep = sm.SE3.Tx(-0.3) * sm.SE3.Ty(0.1) * sm.SE3.Tz(0.3)

gPanda.base = sm.SE3(0, 0.5, 0)
qPanda.base = sm.SE3(0, -0.5, 0)

gTep = gPanda.fkine() * eTep
qTep = qPanda.fkine() * eTep

# Gain term (lambda) for control minimisation
Y = 0.0005

# Quadratic component of objective function
Q = Y * np.eye(7)

arrived1 = False
arrived2 = False
env.add(gPanda)
env.add(qPanda)
env.add(gPanda.vellipse(centre='ee'))
env.add(qPanda.vellipse(centre='ee'))

dt = 0.01

while not arrived1 and not arrived2:

    start = time.time()
    v1, arrived1 = rp.p_servo(gPanda.fkine(), gTep, 1)
    v2, arrived2 = rp.p_servo(qPanda.fkine(), qTep, 1)

    null = np.linalg.pinv(gPanda.jacobe()) @ gPanda.jacobe() @  gPanda.jacobm()
    null2 = np.linalg.pinv(qPanda.jacobe()) @ qPanda.jacobe() @  qPanda.jacobm()

    gPanda.qd = np.linalg.pinv(gPanda.jacobe()) @ v1 + null.flatten()

    Aeq = qPanda.jacobe()
    beq = v2.reshape((6,))
    c = -qPanda.jacobm().reshape((7,))
    qPanda.qd = qp.solve_qp(Q, c, None, None, Aeq, beq) + null2.flatten()

    env.step(dt * 1000)
    stop = time.time()

    # if stop - start < dt:
    #     time.sleep(dt - (stop - start))

    print("gm: {0}".format(np.round(gPanda.manipulability(), 4)))
    print("qm: {0}".format(np.round(qPanda.manipulability(), 4)))
    print()

# Uncomment to stop the plot from closing
env.hold()
