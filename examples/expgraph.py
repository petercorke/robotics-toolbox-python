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
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 4.5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 1.5
matplotlib.rcParams['axes.labelpad'] = 1
plt.rc('grid', linestyle="-", color='#dbdbdb')


wTe = sm.SE3(np.array([
    [ 0.99908232, -0.00979103,  0.04169728,  0.3857763 ],
    [-0.0106737,  -0.99972252,  0.02099873, -0.14162885],
    [ 0.04148011, -0.02142452, -0.9989096,   0.56963678],
    [ 0,           0,           0,           1,        ]]), check=False)


wTep = sm.SE3(np.array([
    [ 0.9987996,   0.03844678,  0.03035121,  0.50746105],
    [ 0.03559668, -0.99535677,  0.08943028,  0.43880564],
    [ 0.03364859, -0.08824252, -0.99553053,  0.30606948],
    [ 0,           0,           0,           1,        ]]), check=False)

wTs = sm.SE3(np.array([
    [ 0.49507983, -0.12231131,  0.86019527,  0.60358311],
    [-0.14425519, -0.98787197, -0.05744054, -0.07014726],
    [ 0.85678843, -0.09564998, -0.50671952,  0.84229899],
    [ 0,           0,           0,           1,        ]]), check=False)

wTinit = sm.SE3(np.array([
    [ 0.99983599, -0.01805617, -0.00140186,  0.18252277],
    [-0.01770909, -0.99094841,  0.13307002,  0.31792092],
    [-0.00379191, -0.13302337, -0.99110565,  0.49814723],
    [ 0,           0,           0,           1,        ]]), check=False)

# qs = np.array([-0.00854625, -0.97162125, -1.36328456, -1.66128817, -0.89608932,
    # 1.55605436,  1.03810812])

qs = np.array([-0.20738953, -0.0548877 , -1.24075604, -1.75235347, -0.03584731,
    1.74775206, -0.4801399 ])



qinit = np.array([0.01001354, -0.94196696,  0.666929,   -2.35186282,  0.64417565,  1.5923856, 1.26946392])


qe = np.array([-0.36802377, -0.46035445,  0.00213448, -2.03919368,  0.03621091,  1.60855928, 0.42849982])
qe2 = np.array([-0.76802377, -0.46035445,  0.00213448, -2.03919368,  0.03621091,  1.60855928, 0.42849982])

q_e3 = np.array([ 0.72753865,  0.3939903 , -0.02553072, -1.72200977,  0.06517471,
    2.19448239,  1.41855741])

q_e6 = np.array([ 0.57937697,  0.30853719, -0.02962658, -1.38546566,  0.011491  ,
    1.79103358,  1.41999518])

q_e7 = np.array([ 0.58136559, -0.47102934,  0.2124923 , -2.17903279,  0.12434963,
    1.77409287,  0.98258544])


wT_e3 = sm.SE3(np.array([[ 0.51728094, -0.15414056,  0.84182012,  0.4160503 ],
    [ 0.04057563, -0.97812317, -0.20403107,  0.02054094],
    [ 0.85485323,  0.13969877, -0.49971012,  0.98886652],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

# wT_e4 = sm.SE3(np.array([[ 0.95775419, -0.21400353, -0.19211821,  0.40808475],
#     [ 0.20259281,  0.97619716, -0.07742901, -0.22218317],
#     [ 0.20411533,  0.03523619,  0.9783125 ,  0.7300475 ],
#     [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e4 = sm.SE3(np.array([[ 0.95775419, -0.21400353, -0.19211821,  0.30808475],
    [ 0.20259281,  0.97619716, -0.07742901, -0.32218317],
    [ 0.20411533,  0.03523619,  0.9783125 ,  0.7300475 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e5 = sm.SE3(np.array([[ 0.66463707, -0.53480389, -0.5217685 , -0.10688366],
    [-0.73521985, -0.34374666, -0.58420031, -0.32759363],
    [ 0.13307642,  0.77189574, -0.62166521,  0.5877783 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e6 = sm.SE3(np.array([[ 0.99847581, -0.00816676, -0.05458349,  0.36802996],
    [-0.01046091, -0.99906799, -0.04187734,  0.3983214 ],
    [-0.05419062,  0.0423845 , -0.99763066,  0.10721047],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e7 = sm.SE3(np.array([[ 0.99954374, -0.0274862 ,  0.01252281,  0.52575806],
    [-0.02733123, -0.99954974, -0.01238256, -0.16837214],
    [ 0.01285752,  0.01203465, -0.99984491,  0.08254964],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)

wT_e8 = sm.SE3(np.array([[ 0.99938713,  0.0239239 , -0.02555423,  0.39872703],
    [ 0.0240422 , -0.99970156,  0.00433234,  0.03783871],
    [-0.02544296, -0.00494406, -0.99966405,  0.10345876],
    [ 0.        ,  0.        ,  0.        ,  1.        ]]), check=False)








class Exp(object):

    def __init__(self):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(2.5, 1.2)
        
        # self.ax.set_facecolor('white')
        self.ax.set(xlabel='Time (s)', ylabel='Manipulability')
        self.ax.grid()
        plt.grid(True)
        self.ax.set_xlim(xmin=0, xmax=3.1)
        self.ax.set_ylim(ymin=0, ymax=0.11)
        plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)
        # self.ax.autoscale(enable=True, axis='both', tight=False)

        plt.ion()
        plt.show()

        # Plot the robot links
        self.rrm = self.ax.plot(
            [0], [0], label='RRMC')  #, color='#E16F6D')

        # Plot the robot links
        self.gpm = self.ax.plot(
            [0], [0], label='Park [5]')  #, color='#E16F6D')

        # Plot the robot links
        self.qua = self.ax.plot(
            [0], [0], label='MMC (ours)')  #, color='#E16F6D')

        self.ax.legend()
        self.ax.legend(loc="lower right")

        plt.pause(0.1)

        # self.ax.set_xlim([limits[0], limits[1]])
        # self.ax.set_ylim([limits[2], limits[3]])

        # Make Robots
        self.r0 = rp.models.Panda()
        self.r0.name = 'rrm'

        self.r1 = rp.models.Panda()
        self.r1.name = 'gpm'

        self.r2 = rp.models.Panda()
        self.r2.name = 'qua'

        self.r3 = rp.models.Panda()
        self.r3.name = 'new'

        self.r = [self.r0, self.r1, self.r2, self.r3]

        # Set joint limits
        self.n = 7
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
        self.ms = 0.05
        self.itmax = 100

    def step(self, action):
        # Step the rrmc robot
        if not self.r0.arrived and not self.r0.fail:
            try:
                self.step_r(self.r0, self.T0)
            except np.linalg.LinAlgError:
                self.r0.fail = True
                self.r0.s = True
        else:
            self.r0.qd = np.zeros(self.n)

        # Step the gpm robot
        if not self.r1.arrived and not self.r1.fail:
            try:
                self.step_g(self.r1, self.T1)
            except np.linalg.LinAlgError:
                self.r1.fail = True
                self.r1.s = True
        else:
            self.r1.qd = np.zeros(self.n)

        # Step the quadratic robot
        if not self.r2.arrived and not self.r2.fail:
            try:
                self.step_q(self.r2, self.T2)
            except np.linalg.LinAlgError:
                self.r2.fail = True
                self.r2.s = True
        else:
            self.r2.qd = np.zeros(self.n)

        # Step the new robot
        if not self.r3.arrived and not self.r3.fail:
            try:
                self.step_n(self.r3, self.T3)
            except np.linalg.LinAlgError:
                self.r3.fail = True
                self.r3.s = True
        else:
            self.r3.qd = np.zeros(self.n)

        # Step the environment
        self.env.step(self.dt)
        self.it += 1
        plt.pause(0.001)

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
        if rarr == 4:
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
        q_init = np.r_[qinit, 0, 0]
        for r in self.r:
            r.q = q_init.copy()
            r.qd = np.zeros(self.n)
            r.it = 0
            r.s = False
            r.m = 0
            r.arrived = False
            r.fail = False
            r.manip = []

        self.done = False
        self.it = 0


        # Set desired poses
        # self.T = self._find_pose(self.r)
        # self.T = sm.SE3([self.r[0].base * wTs, self.r[1].base * wTs, self.r[2].base * wTs, self.r[3].base * wTs], check=False)
        self.T0 = sm.SE3(self.r[0].base * wTs, check=False)
        self.T1 = sm.SE3(self.r[1].base * wTs, check=False)
        self.T2 = sm.SE3(self.r[2].base * wTs, check=False)
        self.T3 = sm.SE3(self.r[3].base * wTs, check=False)

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
        robot.manip.append(m)

        x = np.linspace(0, (robot.it * self.ms), num=robot.it)
        self.gpm[0].set_xdata(x)
        self.gpm[0].set_ydata(robot.manip)

        if self._check_limit(robot):
            robot.fail = True

    def step_q(self, robot, Ts):
        ps = 0.05
        pi = 0.9

        e, m, _ = self.state(robot, Ts)
        v, robot.arrived = rp.p_servo(robot.fkine(), Ts, 1.05, threshold=0.17)
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
        robot.manip.append(m)

        x = np.linspace(0, (robot.it * self.ms), num=robot.it)
        self.qua[0].set_xdata(x)
        self.qua[0].set_ydata(robot.manip)

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

        robot.qd = np.linalg.pinv(robot.jacobe()) @ v + 1 / Y * null.flatten()

        # if np.any(np.isnan(qd)):
        #     robot.fail = True
        #     robot.s = True
        #     robot.qd = robot.qz
        # else:
        #     robot.qd = qd[:self.n]

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
        robot.manip.append(m)

        x = np.linspace(0, (robot.it * self.ms), num=robot.it)
        self.rrm[0].set_xdata(x)
        self.rrm[0].set_ydata(robot.manip)

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
        q = np.r_[self._rand_q(), 0, 0]
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

    for i in range(1):
        e.reset()
        print()
        print('Interation: {0}'.format(i))
        envfun(e)

    e.fig.savefig('x.eps')

plt.ioff()

plt.show()