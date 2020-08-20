#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import spatialmath as sm
import numpy as np
import time
import random
import qpsolvers as qp


ur = rp.UR5()

# pQuad.name = 'Quad Panda'
# pProj = rp.PandaURDF()
# pProj.name = 'Proj Panda'
# pRrmc = rp.PandaURDF()
# pRrmc.name = 'RRMC Panda'
# qlim = pProj.qlim.copy()

# rang = np.abs(qlim[0, :]) + np.abs(qlim[1, :])


# ur.q = ur.qr
ur.q = [0, -np.pi/2.4, np.pi/2.4, 0, np.pi/2, 0]
# pProj.q = pProj.qr
# pRrmc.q = pRrmc.qr
# pQuad.base = sm.SE3.Ty(0.5)
# pProj.base = sm.SE3.Ty(0)
# pRrmc.base = sm.SE3.Ty(-0.5)

# # Gain term (lambda) for control minimisation
# Y = 0.01

# # Quadratic component of objective function
# Q = Y * np.eye(9)

# def rand_q(k=0.15):
#     q = np.zeros(n)
#     for i in range(n):
#         off = k * rang[i]
#         q[i] = random.uniform(qlim[0, i] + off, qlim[1, i] - off)
#     return q


# def find_pose():
#     q = rand_q()
#     return pQuad.fkine(q), pProj.fkine(q), pRrmc.fkine(q)


# def check_limit(robot):
#     limit = False
#     off = 0.00
#     for i in range(n-2):
#         if robot.q[i] <= (qlim[0, i] + off) or robot.q[i] >= (qlim[1, i] - off):
#             return True

#     return limit


# def mean(fail, m, fm, tot):
#     fq = 0
#     fp = 0
#     fr = 0
#     mmq = 0.0
#     mmp = 0.0
#     mmr = 0
#     mfmq = 0.0
#     mfmp = 0.0
#     mfmr = 0.0
#     j = 0

#     for i in range(tot):
#         if fail[i, 0]:
#             fq += 1
#         if fail[i, 1]:
#             fp += 1
#         if fail[i, 2]:
#             fr += 1

#         if not fail[i, 0] and not fail[i, 1] and not fail[i, 2]:
#             j += 1
#             mmq += m[i, 0]
#             mfmq += fm[i, 0]
#             mmp += m[i, 1]
#             mfmp += fm[i, 1]
#             mmr += m[i, 2]
#             mfmr += fm[i, 2]

#     j = np.max([1, j])
#     mmq = mmq/j
#     mfmq = mfmq/j
#     mmp = mmp/j
#     mfmp = mfmp/j
#     mmr = mmr/j
#     mfmr = mfmr/j

#     print("Quad: fails: {0}, mmean: {1}, mfinal: {2}".format(
#         fq, np.round(mmq, 4), np.round(mfmq, 4)))

#     print("Proj: fails: {0}, mmean: {1}, mfinal: {2}".format(
#         fp, np.round(mmp, 4), np.round(mfmp, 4)))

#     print("RRMC: fails: {0}, mmean: {1}, mfinal: {2}".format(
#         fr, np.round(mmr, 4), np.round(mfmr, 4)))

env = rp.backend.Sim()
env.launch()

# arrivedq = False
# arrivedp = False

env.add(ur)
# env.add(pProj)
# env.add(pRrmc)
# time.sleep(1)

# dt = 0.05

# tests = 10000

# m = np.zeros((tests, 3))
# fm = np.zeros((tests, 3))
# fail = np.zeros((tests, 3))

# for i in range(tests):
#     arrivedq = False
#     arrivedp = False
#     arrivedr = False
#     failq = False
#     failp = False
#     failr = False
#     it = 0
#     mq = 0
#     mp = 0
#     mr = 0

#     q_init = rand_q()

#     pQuad.q = q_init.copy()
#     pProj.q = q_init.copy()
#     pRrmc.q = q_init.copy()
#     pQuad.q = pQuad.qr
#     pProj.q = pProj.qr
#     pRrmc.q = pRrmc.qr

#     pQuad.qd = np.zeros(n)
#     pProj.qd = np.zeros(n)
#     pRrmc.qd = np.zeros(n)
#     env.step(dt)
#     # time.sleep(2)

#     Tq, Tp, Tr = find_pose()
#     # eTep = sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.1) * sm.SE3.Tz(0.3)
#     # Tq = pQuad.fkine() * eTep
#     # Tp = pProj.fkine() * eTep

#     start = time.time()

#     while (
#             (not arrivedq or not arrivedp or not arrivedr) and 
#             (time.time() - start) < tmax and
#             not failq and not failp and not failr):

#         mq += pQuad.manipulability()
#         mp += pProj.manipulability()
#         mr += pRrmc.manipulability()

#         if not arrivedr and not failr:
#             vr, arrivedr = rp.p_servo(pRrmc.fkine(), Tr, 1, threshold=0.1)
#             pRrmc.qd = np.linalg.pinv(pRrmc.jacobe()) @ vr
#         else:
#             pRrmc.qd = np.zeros(n)

#         if not arrivedq and not failq:
#             vq, arrivedq = rp.p_servo(pQuad.fkine(), Tq, 1, threshold=0.1)
#             Aeq = pQuad.jacobe()
#             beq = vq.reshape((6,))
#             c = -pQuad.jacobm().reshape((9,))
#             pQuad.qd = qp.solve_qp(Q, c, None, None, Aeq, beq)
#             # pQuad.qd = np.linalg.pinv(pQuad.jacobe()) @ vq
#         else:
#             pQuad.qd = np.zeros(n)

#         if not arrivedp and not failp:
#             null = (
#                     np.eye(9) - np.linalg.pinv(pProj.jacobe()) @ pProj.jacobe()
#                 ) @ pProj.jacobm()
#             # null = (
#                 # rp.null(pProj.jacobe()) @
#                 # np.linalg.pinv(rp.null(pProj.jacobe())) @
#                 # pProj.jacobm())

#             vp, arrivedp = rp.p_servo(pProj.fkine(), Tp, 1, threshold=0.1)
#             pProj.qd = (np.linalg.pinv(pProj.jacobe()) @ vp) + 100 * null.flatten()
#         else:
#             pProj.qd = np.zeros(n)

#         if check_limit(pRrmc):
#             failr = True

#         if check_limit(pQuad):
#             failq = True

#         if check_limit(pProj):
#             failp = True

#         env.step(dt * 1000)
#         it += 1

#     fail[i, 0] = not arrivedq
#     fail[i, 1] = not arrivedp
#     fail[i, 2] = not arrivedr

#     m[i, 0] = mq / it
#     m[i, 1] = mp / it
#     m[i, 2] = mr / it

#     fm[i, 0] = pQuad.manipulability()
#     fm[i, 1] = pProj.manipulability()
#     fm[i, 2] = pRrmc.manipulability()

#     print("Iteration: {0}".format(i))
#     print("Quad: {0}, mean: {1}, final: {2}".format(
#         arrivedq, np.round(m[i, 0], 4), np.round(fm[i, 0], 4)))

#     print("Proj: {0}, mean: {1}, final: {2}".format(
#         arrivedp, np.round(m[i, 1], 4), np.round(fm[i, 1], 4)))

#     print("Rrmc: {0}, mean: {1}, final: {2}".format(
#         arrivedr, np.round(m[i, 2], 4), np.round(fm[i, 2], 4)))
#     mean(fail, m, fm, i+1)
#     print()
