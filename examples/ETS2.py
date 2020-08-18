#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
# from ropy.backend import xacro
# from ropy.backend import urdf


import spatialmath as sm
import numpy as np
import time
import qpsolvers as qp



env = rp.backend.Sim()
env.launch()

pQuad = rp.PandaURDF()
pProj = rp.PandaURDF()
pQuad.q = pQuad.qr
pProj.q = pQuad.qr
pQuad.base = sm.SE3.Ty(0.4)
pProj.base = sm.SE3.Ty(-0.4)

Tep = pQuad.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(0.4) * sm.SE3.Rx(0.6)* sm.SE3.Ry(0.6)
Tep2 = pProj.fkine() * sm.SE3.Tx(0.2) * sm.SE3.Ty(0.2) * sm.SE3.Tz(-0.4) * sm.SE3.Rx(0.6)* sm.SE3.Ry(0.6)

arrived = False
env.add(pQuad)
env.add(pProj)
time.sleep(1)

dt = 0.05

while not arrived:

    start = time.time()
    v, arrived = rp.p_servo(pQuad.fkine(), Tep, 1)
    v2, _ = rp.p_servo(pProj.fkine(), Tep2, 1)
    pQuad.qd = np.linalg.pinv(pQuad.jacobe()) @ v
    pProj.qd = np.linalg.pinv(pProj.jacobe()) @ v
    env.step(dt * 1000)
    stop = time.time()

# dt = 0.01

# while not arrived1 and not arrived2:

#     start = time.time()
#     v1, arrived1 = rp.p_servo(gPanda.fkine(), gTep, 1)
#     v2, arrived2 = rp.p_servo(qPanda.fkine(), qTep, 1)

#     null = np.linalg.pinv(gPanda.jacobe()) @ gPanda.jacobe() @  gPanda.jacobm()
#     null2 = np.linalg.pinv(qPanda.jacobe()) @ qPanda.jacobe() @  qPanda.jacobm()

#     gPanda.qd = np.linalg.pinv(gPanda.jacobe()) @ v1 + null.flatten()

#     Aeq = qPanda.jacobe()
#     beq = v2.reshape((6,))
#     c = -qPanda.jacobm().reshape((7,))
#     qPanda.qd = qp.solve_qp(Q, c, None, None, Aeq, beq) + null2.flatten()

#     env.step(dt * 1000)
#     stop = time.time()

#     # if stop - start < dt:
#     #     time.sleep(dt - (stop - start))

#     print("gm: {0}".format(np.round(gPanda.manipulability(), 4)))
#     print("qm: {0}".format(np.round(qPanda.manipulability(), 4)))
#     print()

# # Uncomment to stop the plot from closing
# env.hold()




















###########################################

panda = rp.PandaURDF()
# print(panda.fkine(panda.qz))

# for l in panda.ets:
    # for gi in l.geometry:
        # print(gi.filename)

# xacro.main('ropy/models/xarco/panda/panda_arm_hand.urdf.xacro')
