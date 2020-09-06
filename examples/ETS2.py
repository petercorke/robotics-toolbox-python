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

def check_limit(robot):
    limit = False
    off = 0.00
    for i in range(n-2):
        if robot.q[i] <= (qlim[0, i] + off) or robot.q[i] >= (qlim[1, i] - off):
            return True

    return limit


def mean(fail, m, fm, tot):
    fq = 0
    fp = 0
    fr = 0
    mmq = 0.0
    mmp = 0.0
    mmr = 0
    mfmq = 0.0
    mfmp = 0.0
    mfmr = 0.0
    j = 0

    for i in range(tot):
        if fail[i, 0]:
            fq += 1
        if fail[i, 1]:
            fp += 1
        if fail[i, 2]:
            fr += 1

        if not fail[i, 0] and not fail[i, 1] and not fail[i, 2]:
            j += 1
            mmq += m[i, 0]
            mfmq += fm[i, 0]
            mmp += m[i, 1]
            mfmp += fm[i, 1]
            mmr += m[i, 2]
            mfmr += fm[i, 2]

    j = np.max([1, j])
    mmq = mmq/j
    mfmq = mfmq/j
    mmp = mmp/j
    mfmp = mfmp/j
    mmr = mmr/j
    mfmr = mfmr/j

    print("Quad: fails: {0}, mmean: {1}, mfinal: {2}".format(
        fq, np.round(mmq, 4), np.round(mfmq, 4)))

    print("Proj: fails: {0}, mmean: {1}, mfinal: {2}".format(
        fp, np.round(mmp, 4), np.round(mfmp, 4)))

    print("RRMC: fails: {0}, mmean: {1}, mfinal: {2}".format(
        fr, np.round(mmr, 4), np.round(mfmr, 4)))

env = rp.backend.Sim()
env.launch()

arrivedq = False
arrivedp = False

env.add(pQuad)
env.add(pProj)
env.add(pRrmc)
time.sleep(1)

dt = 0.05

tests = 10000

m = np.zeros((tests, 3))
fm = np.zeros((tests, 3))
fail = np.zeros((tests, 3))

for i in range(tests):
    arrivedq = False
    arrivedp = False
    arrivedr = False
    failq = False
    failp = False
    failr = False
    it = 0
    mq = 0
    mp = 0
    mr = 0

    q_init = rand_q()

    pQuad.q = q_init.copy()
    pProj.q = q_init.copy()
    pRrmc.q = q_init.copy()
    pQuad.q = pQuad.qr
    pProj.q = pProj.qr
    pRrmc.q = pRrmc.qr

    pQuad.qd = np.zeros(n)
    pProj.qd = np.zeros(n)
    pRrmc.qd = np.zeros(n)
    env.step(dt)
    # time.sleep(2)

    Tq, Tp, Tr = find_pose()
    # eTep = sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.1) * sm.SE3.Tz(0.3)
    # Tq = pQuad.fkine() * eTep
    # Tp = pProj.fkine() * eTep

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
