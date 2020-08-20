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

n = 6
tmax = 15


urQuad = rp.UR5()
urQuad.name = 'Quad UR5'
# pProj = rp.PandaURDF()
# pProj.name = 'Proj Panda'
urRrmc = rp.UR5()
urRrmc.name = 'RRMC UR5'
qlim = urRrmc.qlim.copy()
rang = np.abs(qlim[0, :]) + np.abs(qlim[1, :])


urRrmc.q = [0, -np.pi/2.4, np.pi/2.4, 0, np.pi/2, 0]
urQuad.q = [0, -np.pi/2.4, np.pi/2.4, 0, np.pi/2, 0]

urQuad.base = sm.SE3.Ty(0.3)
urRrmc.base = sm.SE3.Ty(-0.3)

# Gain term (lambda) for control minimisation
Y = 0.01

# Quadratic component of objective function
Q = np.eye(6 + 6)
Q[:6, :6] *= Y
Q[6:, 6:] *= 10


def rand_q(k=0.15):
    q = np.zeros(n)
    for i in range(n):
        off = k * rang[i]
        q[i] = random.uniform(qlim[0, i] + off, qlim[1, i] - off)

    if np.any(np.isnan(q)):
        print("HEIOJPOIUHPUOHUIH:UOH:U:ON")
    return q


def find_pose():
    q = rand_q()
    return urRrmc.fkine(q), urQuad.fkine(q)


def check_limit(robot):
    limit = False
    off = 0.00
    for i in range(n-2):
        if robot.q[i] <= (qlim[0, i] + off) or robot.q[i] >= (qlim[1, i] - off):
            return True

    return limit


def mean(fail, m, fm, s, tot):
    fq = 0
    # fp = 0
    fr = 0
    mmq = 0.0
    # mmp = 0.0
    mmr = 0
    mfmq = 0.0
    # mfmp = 0.0
    mfmr = 0.0
    sq = 0
    sr = 0
    j = 0

    for i in range(tot):
        if fail[i, 0]:
            fq += 1
        # if fail[i, 1]:
        #     fp += 1
        if fail[i, 2]:
            fr += 1

        if s[i, 2]:
            sr += 1
        if s[i, 0]:
            sq += 1
        if not fail[i, 0] and not fail[i, 1] and not fail[i, 2]:
            j += 1
            mmq += m[i, 0]
            mfmq += fm[i, 0]
            # mmp += m[i, 1]
            # mfmp += fm[i, 1]
            mmr += m[i, 2]
            mfmr += fm[i, 2]

    j = np.max([1, j])
    mmq = mmq/j
    mfmq = mfmq/j
    # mmp = mmp/j
    # mfmp = mfmp/j
    mmr = mmr/j
    mfmr = mfmr/j

    print("Quad: fails: {0}, mmean: {1}, mfinal: {2}, singulars: {3}".format(
        fq, np.round(mmq, 4), np.round(mfmq, 4), sr))

    # print("Proj: fails: {0}, mmean: {1}, mfinal: {2}".format(
    #     fp, np.round(mmp, 4), np.round(mfmp, 4)))

    print("RRMC: fails: {0}, mmean: {1}, mfinal: {2}, singulars: {3}".format(
        fr, np.round(mmr, 4), np.round(mfmr, 4), sr))


env = rp.backend.Sim()
env.launch()

env.add(urRrmc)
# env.add(pProj)
env.add(urQuad)
time.sleep(1)

dt = 0.05

tests = 10000

m = np.zeros((tests, 3))
fm = np.zeros((tests, 3))
fail = np.zeros((tests, 3))
s = np.zeros((tests, 3))

for i in range(tests):
    arrivedq = False
    arrivedr = False
    failq = False
    failr = False
    it = 0
    mq = 0
    mr = 0

    q_init = rand_q()

    urRrmc.q = q_init.copy()
    urQuad.q = q_init.copy()

    urQuad.qd = np.zeros(n)
    urRrmc.qd = np.zeros(n)
    env.step(dt)

    Tr, Tq = find_pose()

    start = time.time()

    while (
            (not arrivedq or not arrivedr) and 
            (time.time() - start) < tmax and
            not failq and not failr):

        mq += urQuad.manipulability()
        mr += urRrmc.manipulability()

        if not arrivedr and not failr:
            try:
                if np.linalg.matrix_rank(urRrmc.jacobe()) < 6:
                    s[i, 2] = True
                    failr = True
                vr, arrivedr = rp.p_servo(urRrmc.fkine(), Tr, 1, threshold=0.1)
                urRrmc.qd = np.linalg.inv(urRrmc.jacobe()) @ vr
            except np.linalg.LinAlgError:
                failr = True
                s[i, 2] = True
        else:
            urRrmc.qd = np.zeros(n)

        if not arrivedq and not failq:
            try:
                vq, arrivedq = rp.p_servo(urQuad.fkine(), Tq, 1, threshold=0.1)

                eTep = urQuad.fkine().inv() * Tq
                e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

                Q[6:, 6:] = (1 / e) * np.eye(6)

                Aeq = np.c_[urQuad.jacobe(), np.eye(6)]

                beq = vq.reshape((6,))
                c = np.r_[-urQuad.jacobm().reshape((6,)), np.zeros(6)]
                qd = qp.solve_qp(Q, c, None, None, Aeq, beq)
                if np.any(np.isnan(qd)):
                    failq = True
                    s[i, 0] = True
                    urQuad.qd = urQuad.qz
                else:
                    urQuad.qd = qd[:6]
            except np.linalg.LinAlgError:
                failq = True
                s[i, 0] = True
        else:
            urQuad.qd = np.zeros(n)

        if check_limit(urRrmc):
            failr = True

        if check_limit(urQuad):
            failq = True

        env.step(dt * 1000)
        it += 1

    fail[i, 0] = not arrivedq
    fail[i, 2] = not arrivedr

    m[i, 0] = mq / it
    m[i, 2] = mr / it

    fm[i, 0] = urQuad.manipulability()
    fm[i, 2] = urRrmc.manipulability()

    print("Iteration: {0}".format(i + 1))
    print("Quad: {0}, mean: {1}, final: {2}, singular: {3}".format(
        arrivedq, np.round(m[i, 0], 4), np.round(fm[i, 0], 4), s[i, 0]))

    print("Rrmc: {0}, mean: {1}, final: {2}, singular: {3}".format(
        arrivedr, np.round(m[i, 2], 4), np.round(fm[i, 2], 4), s[i, 2]))
    mean(fail, m, fm, s, i+1)
    print()
