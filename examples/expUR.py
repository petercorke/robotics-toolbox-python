#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time
import random
import qpsolvers as qp


class Exp(object):

    def __init__(self):

        # Make Robots
        self.r0 = rp.models.UR5()
        self.r0.name = 'rrm'

        self.r1 = rp.models.UR5()
        self.r1.name = 'gpm'

        self.r2 = rp.models.UR5()
        self.r2.name = 'qua'

        self.r3 = rp.models.UR5()
        self.r3.name = 'new'

        self.r = [self.r0, self.r1, self.r2, self.r3]

        # Set joint limits
        self.n = 6
        self.qlim = self.r[0].qlim.copy()
        self.rang = np.abs(self.qlim[0, :]) + np.abs(self.qlim[1, :])

        # Set base locations
        # Init robot variables
        for i in range(len(self.r)):
            self.r[i].base = sm.SE3.Ty(i * 0.9 + 0.3)
            self.r[i].failt = 0
            self.r[i].arrivedt = 0
            self.r[i].s = False
            self.r[i].st = 0
            self.r[i].mt = []
            self.r[i].mft = []
            self.r[i].missed = 0

        # Launch Sim
        self.env = rp.backend.Swift()
        self.env.launch()

        # Add robots
        for r in self.r:
            self.env.add(r)
        time.sleep(1)

        # Timestep
        self.dt = 50
        self.itmax = 100

    def step(self, action):
        # Step the rrmc robot
        if not self.r0.arrived and not self.r0.fail:
            try:
                self.step_r(self.r0, self.T[0])
            except np.linalg.LinAlgError:
                self.r0.fail = True
                self.r0.s = True
        else:
            self.r0.qd = np.zeros(self.n)

        # # Step the gpm robot
        # if not self.r1.arrived and not self.r1.fail:
        #     try:
        #         self.step_g(self.r1, self.T[1])
        #     except np.linalg.LinAlgError:
        #         self.r1.fail = True
        #         self.r1.s = True
        # else:
        #     self.r1.qd = np.zeros(self.n)

        # Step the quadratic robot
        if not self.r2.arrived and not self.r2.fail:
            try:
                self.step_q(self.r2, self.T[2])
            except np.linalg.LinAlgError:
                self.r2.fail = True
                self.r2.s = True
        else:
            self.r2.qd = np.zeros(self.n)

        # # Step the new robot
        # if not self.r3.arrived and not self.r3.fail:
        #     try:
        #         self.step_n(self.r3, self.T[3])
        #     except np.linalg.LinAlgError:
        #         self.r3.fail = True
        #         self.r3.s = True
        # else:
        #     self.r3.qd = np.zeros(self.n)

        # Step the environment
        self.env.step(self.dt)
        self.it += 1

        if ((self.r2.arrived or self.r2.fail)
                and (self.r0.arrived or self.r0.fail)) \
                or self.it > self.itmax:
            self.done = True
            self.finished(self.r)
            self.mean(self.r)
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

            if not robot.arrived and not robot.fail:
                robot.missed += 1

            try:
                m = robot.m / robot.it
                mf = robot.manipulability()
            except:
                m = 0

            print("{0}: {1}, mean: {2}, final: {3}, singular: {4}".format(
                robot.name, robot.arrived, np.round(m, 4),
                np.round(mf, 4), robot.s))

        # if rarr == len(robots):
        if rarr == 2:
            for robot in robots:
                try:
                    robot.mt.append(robot.m / robot.it)
                    robot.mft.append(robot.manipulability())
                except:
                    pass

    def mean(self, robots):
        print("Dual success: {0}".format(len(robots[0].mt)))
        for robot in robots:
            mm = np.sum(robot.mt) / len(robot.mt)
            mmf = np.sum(robot.mft) / len(robot.mft)
            print("{0}: fails: {1}, mmean: {2}, mfinal: {3},"
                  " singulars: {4}, missed: {5}".format(
                      robot.name, robot.failt,
                      np.round(mm, 4), np.round(mmf, 4), robot.st,
                      robot.missed))

    def reset(self):
        # Set initial joint angles
        # q_init = np.r_[self._rand_q(), 0, 0]
        q_init = self._rand_q()
        for r in self.r:
            r.q = q_init.copy()
            r.qd = np.zeros(self.n)
            r.it = 0
            r.s = False
            r.m = 0
            r.arrived = False
            r.fail = False

        self.done = False
        self.it = 0

        # Set desired poses
        self.T = self._find_pose(self.r)

    def step_g(self, robot, Ts):
        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1, threshold=0.17)
        Y = 0.01

        Q = np.eye(self.n) * Y
        Aeq = robot.jacobe()
        beq = v.reshape((6,))
        c = -robot.jacobm().reshape((self.n,))
        qd = qp.solve_qp(Q, c, None, None, Aeq, beq)

        if np.any(np.isnan(qd)):
            robot.fail = True
            robot.s = True
            robot.qd = robot.qz
        else:
            robot.qd = qd[:self.n]

        robot.m += m
        robot.it += 1

        if self._check_limit(robot):
            robot.fail = True

    def step_q(self, robot, Ts):
        ps = 0.05
        pi = 0.9

        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1, threshold=0.17)
        Y = 0.01

        Ain = np.zeros((self.n + 6, self.n + 6))
        bin = np.zeros(self.n + 6)

        for i in range(self.n):
            if robot.q[i] - self.qlim[0, i] <= pi:
                bin[i] = -1.0 * (((self.qlim[0, i] - robot.q[i]) + ps) / (pi - ps))
                Ain[i, i] = -1
            if self.qlim[1, i] - robot.q[i] <= pi:
                bin[i] = ((self.qlim[1, i] - robot.q[i]) - ps) / (pi - ps)
                Ain[i, i] = 1

        Q = np.eye(self.n + 6)
        Q[:self.n, :self.n] *= Y
        Q[self.n:, self.n:] = (1 / e) * np.eye(6)
        Aeq = np.c_[robot.jacobe(), np.eye(6)]
        beq = v.reshape((6,))
        c = np.r_[-robot.jacobm().reshape((self.n,)), np.zeros(6)]
        qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq)

        if np.any(np.isnan(qd)):
            robot.fail = True
            robot.s = True
            robot.qd = robot.qz
        else:
            robot.qd = qd[:self.n]

        robot.m += m
        robot.it += 1

        if self._check_limit(robot):
            robot.fail = True

    def step_n(self, robot, Ts):
        ps = 0.05
        pi = 0.9

        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1, threshold=0.17)
        Y = 0.01

        b = np.zeros((self.n, 1))

        for i in range(self.n):
            if robot.q[i] - self.qlim[0, i] <= pi:
                b[i, 0] = -1 * np.power(((robot.q[i] - self.qlim[0, i]) - pi), 2) / np.power((ps - pi), 2)
            if self.qlim[1, i] - robot.q[i] <= pi:
                b[i, 0] = 1 * np.power(((self.qlim[1, i] - robot.q[i]) - pi), 2) / np.power((ps - pi), 2)

        null = (
            np.eye(self.n) - np.linalg.pinv(robot.jacobe()) @ robot.jacobe()
        ) @ (robot.jacobm() - b)

        qd = np.linalg.pinv(robot.jacobe()) @ v + 1 / Y * null.flatten()

        if np.any(np.isnan(qd)):
            robot.fail = True
            robot.s = True
            robot.qd = robot.qz
        else:
            robot.qd = qd[:self.n]

        robot.m += m
        robot.it += 1

        if self._check_limit(robot):
            robot.fail = True

    def step_r(self, robot, Ts):
        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1, threshold=0.17)

        # if np.linalg.matrix_rank(robot.jacobe()) < 6:
        #     robot.s = True
        #     robot.fail = True

        robot.qd = np.linalg.pinv(robot.jacobe()) @ v
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

    def _rand_q(self, k=0.27):
        q = np.zeros(self.n)
        for i in range(self.n):
            off = k * self.rang[i]
            q[i] = random.uniform(self.qlim[0, i] + off, self.qlim[1, i] - off)

        return q

    def _find_pose(self, robots):
        # q = np.r_[self._rand_q(), 0, 0]
        q = self._rand_q()
        T = robots[0].fkine(q)
        for i in range(1, len(robots)):
            T.append(robots[i].fkine(q))
        return T

    def _check_limit(self, robot):
        limit = False
        off = 0.00
        for i in range(self.n-2):
            if robot.q[i] <= (self.qlim[0, i] + off):
                if robot.name == "Quad UR5":
                    print(str(robot.q[i]) + " below ---------------------------------------------------------")
                return True
            elif robot.q[i] >= (self.qlim[1, i] - off):
                if robot.name == "Quad UR5":
                    print(str(robot.q[i]) + " above ---------------------------------------------------------")
                return True

        return limit


def envfun(e):

    while not e.done:
        e.step(0)


if __name__ == '__main__':

    e = Exp()

    for i in range(100000):
        e.reset()
        print()
        print('Interation: {0}'.format(i))
        envfun(e)
