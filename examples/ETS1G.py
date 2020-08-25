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

# n = 6
# tmax = 15


# urQuad = rp.UR5()
# urQuad.name = 'Quad UR5'
# # pProj = rp.PandaURDF()
# # pProj.name = 'Proj Panda'
# urRrmc = rp.UR5()
# urRrmc.name = 'RRMC UR5'
# qlim = urRrmc.qlim.copy()
# rang = np.abs(qlim[0, :]) + np.abs(qlim[1, :])


# urRrmc.q = [0, -np.pi/2.4, np.pi/2.4, 0, np.pi/2, 0]
# urQuad.q = [0, -np.pi/2.4, np.pi/2.4, 0, np.pi/2, 0]

# urQuad.base = sm.SE3.Ty(0.3)
# urRrmc.base = sm.SE3.Ty(-0.3)

# Gain term (lambda) for control minimisation
# Y = 0.01

# Quadratic component of objective function
# Q = np.eye(6 + 6)
# Q[:6, :6] *= Y
# Q[6:, 6:] *= 10



# def mean(fail, m, fm, s, tot):
#     fq = 0
#     # fp = 0
#     fr = 0
#     mmq = 0.0
#     # mmp = 0.0
#     mmr = 0
#     mfmq = 0.0
#     # mfmp = 0.0
#     mfmr = 0.0
#     sq = 0
#     sr = 0
#     j = 0

#     for i in range(tot):
#         if fail[i, 0]:
#             fq += 1
#         # if fail[i, 1]:
#         #     fp += 1
#         if fail[i, 2]:
#             fr += 1

#         if s[i, 2]:
#             sr += 1
#         if s[i, 0]:
#             sq += 1
#         if not fail[i, 0] and not fail[i, 1] and not fail[i, 2]:
#             j += 1
#             mmq += m[i, 0]
#             mfmq += fm[i, 0]
#             # mmp += m[i, 1]
#             # mfmp += fm[i, 1]
#             mmr += m[i, 2]
#             mfmr += fm[i, 2]

#     j = np.max([1, j])
#     mmq = mmq/j
#     mfmq = mfmq/j
#     # mmp = mmp/j
#     # mfmp = mfmp/j
#     mmr = mmr/j
#     mfmr = mfmr/j

#     print("Quad: fails: {0}, mmean: {1}, mfinal: {2}, singulars: {3}".format(
#         fq, np.round(mmq, 4), np.round(mfmq, 4), sr))

#     # print("Proj: fails: {0}, mmean: {1}, mfinal: {2}".format(
#     #     fp, np.round(mmp, 4), np.round(mfmp, 4)))

#     print("RRMC: fails: {0}, mmean: {1}, mfinal: {2}, singulars: {3}".format(
#         fr, np.round(mmr, 4), np.round(mfmr, 4), sr))


# env = rp.backend.Sim()
# env.launch()

# env.add(urRrmc)
# # env.add(pProj)
# env.add(urQuad)
# time.sleep(1)

# dt = 0.05

# tests = 10000

# m = np.zeros((tests, 3))
# fm = np.zeros((tests, 3))
# fail = np.zeros((tests, 3))
# s = np.zeros((tests, 3))

# for i in range(tests):
    # arrivedq = False
    # arrivedr = False
    # failq = False
    # failr = False
    # it1 = 0
    # it2 = 0
    # mq = 0
    # mr = 0

    # q_init = rand_q()

    # urRrmc.q = q_init.copy()
    # urQuad.q = q_init.copy()

    # urQuad.qd = np.zeros(n)
    # urRrmc.qd = np.zeros(n)
    # env.step(dt)

    # Tr, Tq = find_pose()

    # start = time.time()

    # while (
    #         (
    #             (not arrivedq and not failq) or
    #             (not arrivedr and not failr)
    #         ) and (time.time() - start) < tmax):

    #     if not arrivedr and not failr:
    #         try:
    #             mq += urQuad.manipulability()
    #             if np.linalg.matrix_rank(urRrmc.jacobe()) < 6:
    #                 s[i, 2] = True
    #                 failr = True
    #             vr, arrivedr = rp.p_servo(urRrmc.fkine(), Tr, 1, threshold=0.1)
    #             urRrmc.qd = np.linalg.inv(urRrmc.jacobe()) @ vr
    #             it1 += 1
    #         except np.linalg.LinAlgError:
    #             failr = True
    #             s[i, 2] = True
    #     else:
    #         urRrmc.qd = np.zeros(n)

        # if not arrivedq and not failq:
        #     try:
        #         mr += urRrmc.manipulability()
        #         vq, arrivedq = rp.p_servo(urQuad.fkine(), Tq, 1, threshold=0.1)

        #         eTep = urQuad.fkine().inv() * Tq
        #         e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

        #         Q[6:, 6:] = (1 / e) * np.eye(6)

        #         Aeq = np.c_[urQuad.jacobe(), np.eye(6)]

        #         beq = vq.reshape((6,))
        #         c = np.r_[-urQuad.jacobm().reshape((6,)), np.zeros(6)]
        #         qd = qp.solve_qp(Q, c, None, None, Aeq, beq)
        #         if np.any(np.isnan(qd)):
        #             failq = True
        #             s[i, 0] = True
        #             urQuad.qd = urQuad.qz
        #         else:
        #             urQuad.qd = qd[:6]
        #         it2 += 1
        #     except np.linalg.LinAlgError:
        #         failq = True
        #         s[i, 0] = True
        # else:
        #     urQuad.qd = np.zeros(n)

    #     if check_limit(urRrmc):
    #         failr = True

    #     if check_limit(urQuad):
    #         failq = True

    #     env.step(dt * 1000)

    # fail[i, 0] = not arrivedq
    # fail[i, 2] = not arrivedr

    # m[i, 0] = mq / it2
    # m[i, 2] = mr / it1

    # fm[i, 0] = urQuad.manipulability()
    # fm[i, 2] = urRrmc.manipulability()

    # print("Iteration: {0}".format(i + 1))
    # print("Quad: {0}, mean: {1}, final: {2}, singular: {3}".format(
    #     arrivedq, np.round(m[i, 0], 4), np.round(fm[i, 0], 4), s[i, 0]))

    # print("Rrmc: {0}, mean: {1}, final: {2}, singular: {3}".format(
    #     arrivedr, np.round(m[i, 2], 4), np.round(fm[i, 2], 4), s[i, 2]))
    # mean(fail, m, fm, s, i+1)
    # print()


class Exp(object):

    def __init__(self):

        # Make Robots
        self.rQ = rp.UR5()
        self.rQ.name = 'Quad UR5'

        self.rR = rp.UR5()
        self.rR.name = 'RRMC UR5'

        # Set joint limits
        self.n = self.rQ.n
        self.qlim = self.rQ.qlim.copy()
        self.rang = np.abs(self.qlim[0, :]) + np.abs(self.qlim[1, :])

        # Set base locations
        self.rQ.base = sm.SE3.Ty(0.3)
        self.rR.base = sm.SE3.Ty(-0.3)

        # Init robot variables
        self.rQ.failt = 0
        self.rR.failt = 0
        self.rQ.arrivedt = 0
        self.rR.arrivedt = 0
        self.rQ.s = False
        self.rR.s = False
        self.rQ.st = 0
        self.rR.st = 0
        self.rQ.mt = []
        self.rR.mt = []
        self.rQ.mft = []
        self.rR.mft = []

        # Launch Sim
        self.env = rp.backend.Sim()
        self.env.launch()

        # Add robots
        self.env.add(self.rQ)
        self.env.add(self.rR)
        time.sleep(1)

        # Timestep
        self.dt = 50
        self.itmax = 500

    def step(self, action):
        # Step the quadratic robot
        if not self.rQ.arrived and not self.rQ.fail:
            try:
                self.step_q(self.rQ, self.TQ)
            except np.linalg.LinAlgError:
                self.rQ.fail = True
                self.rQ.s = True
        else:
            self.rQ.qd = np.zeros(self.n)

        # Step the rrmc robot
        if not self.rR.arrived and not self.rR.fail:
            try:
                self.step_r(self.rR, self.TR)
            except np.linalg.LinAlgError:
                self.rR.fail = True
                self.rR.s = True
        else:
            self.rR.qd = np.zeros(self.n)

        # Step the environment
        self.env.step(self.dt)
        self.it += 1

        if ((self.rQ.arrived or self.rQ.fail)
                and (self.rR.arrived or self.rR.fail)) \
                or self.it > self.itmax:
            self.done = True
            self.finished([self.rQ, self.rR])
            self.mean([self.rQ, self.rR])
        else:
            self.done = False

    def finished(self, robots):

        rarr = 0
        for robot in robots:
            if robot.arrived:
                robot.arrivedt += 1
                rarr += 1

            elif robot.fail:
                robot.failt += 1

            if robot.s:
                robot.st += 1

            m = robot.m / robot.it
            mf = robot.manipulability()

            print("{0}: {1}, mean: {2}, final: {3}, singular: {4}".format(
                robot.name, robot.arrived, np.round(m, 4),
                np.round(mf, 4), robot.s))

        if rarr == len(robots):
            for robot in robots:
                robot.mt.append(robot.m / robot.it)
                robot.mft.append(robot.manipulability())

    def mean(self, robots):

        for robot in robots:
            mm = np.sum(robot.mt) / len(robot.mt)
            mmf = np.sum(robot.mft) / len(robot.mft)
            print("{0}: fails: {1}, mmean: {2}, mfinal: {3},"
                  " singulars: {4}".format(
                      robot.name, robot.failt,
                      np.round(mm, 4), np.round(mmf, 4), robot.st))

    def reset(self):
        # Set initial joint angles
        q_init = self._rand_q()
        self.rQ.q = q_init.copy()
        self.rR.q = q_init.copy()

        # Set joint velocities to 0
        self.rQ.qd = np.zeros(self.n)
        self.rR.qd = np.zeros(self.n)

        # Robot stats
        self.rQ.it = 0
        self.rQ.s = False
        self.rQ.m = 0
        self.rQ.arrived = False
        self.rQ.fail = False
        self.rR.it = 0
        self.rR.m = 0
        self.rR.arrived = False
        self.rR.fail = False
        self.rR.s = False

        self.done = False
        self.it = 0

        # Set desired poses
        self.TQ, self.TR = self._find_pose((self.rQ, self.rR))

    def step_q(self, robot, Ts):

        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1, threshold=0.17)
        Y = 0.01

        Q = np.eye(6 + 6)
        Q[:6, :6] *= Y
        Q[6:, 6:] = (1 / e) * np.eye(6)
        Aeq = np.c_[robot.jacobe(), np.eye(6)]
        beq = v.reshape((6,))
        c = np.r_[-robot.jacobm().reshape((6,)), np.zeros(6)]
        qd = qp.solve_qp(Q, c, None, None, Aeq, beq)

        if np.any(np.isnan(qd)):
            robot.fail = True
            robot.s = True
            robot.qd = robot.qz
        else:
            robot.qd = qd[:6]

        robot.m += m
        robot.it += 1

        if self._check_limit(robot):
            robot.fail = True

    def step_r(self, robot, Ts):
        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1, threshold=0.17)

        if np.linalg.matrix_rank(robot.jacobe()) < 6:
            robot.s = True
            robot.fail = True

        robot.qd = np.linalg.inv(robot.jacobe()) @ v
        robot.m += m
        robot.it += 1

        if self._check_limit(robot):
            robot.fail = True

    def state(self, robot, Ts):
        arrived = False
        eTep = robot.fkine().inv() * Ts
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))
        m = robot.manipulability()

        if e < 0.1:
            arrived = True
        return e, m, arrived

    def restart(self):
        pass

    def render(self):
        pass

    def _rand_q(self, k=0.15):
        q = np.zeros(self.n)
        for i in range(self.n):
            off = k * self.rang[i]
            q[i] = random.uniform(self.qlim[0, i] + off, self.qlim[1, i] - off)

        return q

    def _find_pose(self, robots):
        q = self._rand_q()
        return robots[0].fkine(q), robots[1].fkine(q)

    def _check_limit(self, robot):
        limit = False
        off = 0.00
        for i in range(self.n-2):
            if (robot.q[i] <= (self.qlim[0, i] + off)
               or robot.q[i] >= (self.qlim[1, i] - off)):
                return True

        return limit


def envfun(e):

    while not e.done:
        e.step(0)
        # print()
        # print(e.rQ.arrived)
        # print(e.rQ.fail)
        # print(e.rQ.qd)
        # print(e.rR.arrived)
        # print(e.rR.fail)
        # print(e.rR.qd)


if __name__ == '__main__':

    e = Exp()

    for i in range(1000):
        e.reset()
        print()
        print('Interation: {0}'.format(i))
        envfun(e)
