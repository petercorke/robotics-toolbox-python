#!/usr/bin/env python3

import unittest

try:
    from bdsim import BDSim
except ModuleNotFoundError:
    raise unittest.SkipTest("bdsim not found, skipping all tests in test_blocks.py") from None

from spatialmath import SE3
from spatialmath.base import tr2x

import numpy.testing as nt

import roboticstoolbox as rtb
from roboticstoolbox.blocks import *
from roboticstoolbox.blocks.quad_model import quadrotor

class State:
    T = 5

    class Opt:
        def __init__(self):
            self.graphics = True
            self.animation = False

    def __init__(self):
        self.options = self.Opt()

class RobotBlockTest(unittest.TestCase):
    def test_fkine(self):

        robot = rtb.models.ETS.Panda()
        q = robot.configs["qr"]
        T = robot.fkine(q)

        block = FKine(robot)
        nt.assert_array_almost_equal(block.T_output(q)[0], T)

    def test_ikine(self):

        robot = rtb.models.ETS.Panda()
        q = robot.configs["qr"]
        T = robot.fkine(q)
        sol = robot.ikine_LM(T)

        block = IKine(robot, seed=0)

        q_ik = block.T_output(T)[0]  # get IK from block
        pass
        nt.assert_array_almost_equal(robot.fkine(q_ik), T)  # test it's FK is correct

    def test_jacobian(self):

        robot = rtb.models.ETS.Panda()
        q = robot.configs["qr"]

        J = robot.jacob0(q)
        block = Jacobian(robot)
        nt.assert_array_almost_equal(block.T_output(q)[0], J)

        J = robot.jacobe(q)
        block = Jacobian(robot, frame="e")
        nt.assert_array_almost_equal(block.T_output(q)[0], J)

        J = robot.jacob0(q)
        block = Jacobian(robot, pinv=True)
        nt.assert_array_almost_equal(block.T_output(q)[0], np.linalg.pinv(J))

    def test_gravload(self):

        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = Gravload(robot)
        nt.assert_array_almost_equal(block.T_output(q)[0], robot.gravload(q))

    def test_gravload_x(self):
        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = Gravload_X(robot)
        nt.assert_array_almost_equal(block.T_output(q)[0], robot.gravload_x(q))

    def test_inertia(self):
        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = Inertia(robot)
        nt.assert_array_almost_equal(block.T_output(q)[0], robot.inertia(q))

    def test_inertia_x(self):
        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = Inertia_X(robot)
        nt.assert_array_almost_equal(block.T_output(q)[0], robot.inertia_x(q))

    def test_idyn(self):
        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = IDyn(robot)
        qd = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        qdd = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        nt.assert_array_almost_equal(
            block.T_output(q, qd, qdd)[0], robot.rne(q, qd, qdd)
        )

        block = IDyn(robot, gravity=[0, 0, 0])
        qd = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        qdd = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        nt.assert_array_almost_equal(
            block.T_output(q, qd, qdd)[0], robot.rne(q, qd, qdd, gravity=[0, 0, 0])
        )

    def test_fdyn(self):
        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = FDyn(robot, q)
        qd = np.zeros((6,))
        tau = [6, 5, 4, 3, 2, 1]

        x = np.r_[q, np.zeros((6,))]
        xd = np.r_[np.zeros((6,)), robot.accel(q, qd, tau)]
        nt.assert_equal(block.T_deriv(tau, x=x), xd)
        nt.assert_equal(block.T_output(tau, x=x)[0], q)
        nt.assert_equal(block.getstate0(), x)

    @unittest.skip
    def test_fdyn_x(self):
        robot = rtb.models.DH.Puma560()
        q = robot.configs["qn"]

        block = FDyn_X(robot, q)
        qd = np.zeros((6,))
        tau = [6, 5, 4, 3, 2, 1]

        x = np.r_[tr2x(robot.fkine(q).A), np.zeros((6,))]
        xd = np.r_[np.zeros((6,)), robot.accel_x(q, qd, tau)]
        block.test_inputs = [tau]  # set inputs
        block._x = x  # set state [x, xd]
        nt.assert_equal(block.deriv(), xd)
        nt.assert_equal(block.output()[0], tr2x(robot.fkine(q).A))
        nt.assert_equal(block.getstate0(), x)

    @unittest.skip("cant test bdsim plot blocks")
    def test_armplot(self):

        robot = rtb.models.ETS.Panda()
        q = robot.configs["qr"]

        block = ArmPlot(robot)


class SpatialBlockTest(unittest.TestCase):
    def test_delta(self):

        block = Tr2Delta()

        T1 = SE3()
        T2 = SE3.Trans(0.01, 0.02, 0.03) * SE3.RPY(0.01, 0.02, 0.03)
        nt.assert_array_almost_equal(block.T_output(T1, T2)[0], T1.delta(T2))

        block = Delta2Tr()

        delta = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        nt.assert_array_almost_equal(block.T_output(delta)[0], SE3.Delta(delta))

    def test_tr2t(self):

        T = SE3.Trans(1, 2, 3) * SE3.RPY(0.3, 0.4, 0.5)

        block = TR2T()
        out = block.T_output(T)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0], 1)
        self.assertAlmostEqual(out[1], 2)
        self.assertAlmostEqual(out[2], 3)

    def test_point2tr(self):

        T = SE3.Trans(1, 2, 3) * SE3.RPY(0.3, 0.4, 0.5)

        block = Point2Tr(T)

        t = np.r_[3, 4, 5]
        nt.assert_array_almost_equal(
            block.T_output(t)[0], SE3.Trans(t) * SE3.RPY(0.3, 0.4, 0.5)
        )

    @unittest.skip
    def test_jtraj(self):
        robot = rtb.models.DH.Puma560()
        q1 = robot.configs["qz"]
        q2 = robot.configs["qr"]

        block = JTraj(q1, q2)
        s = State()
        block.start(s)
        nt.assert_array_almost_equal(block.T_output(t=0)[0], q1)
        nt.assert_array_almost_equal(block.T_output(t=5)[0], q2)

    def test_ctraj(self):

        T1 = SE3.Trans(1, 2, 3) * SE3.RPY(0.3, 0.4, 0.5)
        T2 = SE3.Trans(-1, -2, -3) * SE3.RPY(-0.3, -0.4, -0.5)

        block = CTraj(T1, T2, T=5)

        s = State()
        block.start(s)

        nt.assert_array_almost_equal(block.T_output(t=0)[0], T1)
        nt.assert_array_almost_equal(block.T_output(t=5)[0], T2)

    def test_trapezoidal(self):

        block = Trapezoidal(2, 3, T=5)

        s = State()
        block.start(s)

        out = block.T_output(t=0)
        nt.assert_array_almost_equal(out[0], 2)
        nt.assert_array_almost_equal(out[1], 0)

        out = block.T_output(t=5)
        nt.assert_array_almost_equal(out[0], 3)
        nt.assert_array_almost_equal(out[1], 0)

    def test_circlepath(self):

        block = CirclePath(
            radius=2, centre=[1, 2, 3], frequency=0.25, phase=0, unit="rps"
        )

        nt.assert_array_almost_equal(block.T_output(t=0)[0], (1 + 2, 2, 3))
        nt.assert_array_almost_equal(block.T_output(t=1)[0], (1, 2 + 2, 3))
        nt.assert_array_almost_equal(block.T_output(t=2)[0], (1 - 2, 2, 3))

    def test_traj(self):

        block = Traj([1, 2], [3, 4], time=True, traj="trapezoidal", T=5)
        s = State()
        block.start(s)

        nt.assert_array_almost_equal(block.T_output(t=0)[0], [1, 2])
        nt.assert_array_almost_equal(block.T_output(t=0)[1], [0, 0])

        nt.assert_array_almost_equal(block.T_output(t=5)[0], [3, 4])
        nt.assert_array_almost_equal(block.T_output(t=5)[1], [0, 0])

        nt.assert_array_almost_equal(block.T_output(t=2.5)[0], [2, 3])

        block = Traj([1, 2], [3, 4], time=True, traj="quintic", T=5)
        block.start(s)

        nt.assert_array_almost_equal(block.T_output(t=0)[0], [1, 2])
        nt.assert_array_almost_equal(block.T_output(t=0)[1], [0, 0])
        nt.assert_array_almost_equal(block.T_output(t=0)[2], [0, 0])

        nt.assert_array_almost_equal(block.T_output(t=5)[0], [3, 4])
        nt.assert_array_almost_equal(block.T_output(t=5)[1], [0, 0])
        nt.assert_array_almost_equal(block.T_output(t=5)[2], [0, 0])

        nt.assert_array_almost_equal(block.T_output(t=2.5)[0], [2, 3])


class MobileBlockTest(unittest.TestCase):
    def test_bicycle(self):

        x = [2, 3, np.pi / 2]
        block = Bicycle(x0=x, L=3)

        nt.assert_array_almost_equal(block.T_output(0, 0, x=x, t=0)[0], x)
        nt.assert_array_almost_equal(block.T_deriv(0, 0, x=x), [0, 0, 0])

        nt.assert_array_almost_equal(block.T_output(10, 0.3, x=x, t=0)[0], x)
        nt.assert_array_almost_equal(
            block.T_deriv(10, 0.3, x=x), [10 * np.cos(x[2]), 10 * np.sin(x[2]), 10 / 3 * np.tan(0.3)]
        )

    def test_unicycle(self):

        x = [2, 3, np.pi / 2]
        block = Unicycle(x0=x, W=3)

        nt.assert_array_almost_equal(block.T_output(0, 0, x=x)[0], x)
        nt.assert_array_almost_equal(block.T_deriv(0, 0, x=x), [0, 0, 0])

        nt.assert_array_almost_equal(block.T_output(10, 0.3, x=x)[0], x)
        nt.assert_array_almost_equal(
            block.T_deriv(10, 0.3, x=x), [10 * np.cos(x[2]), 10 * np.sin(x[2]), 0.3]
        )

    def test_diffsteer(self):

        x = [2, 3, np.pi / 2]
        block = DiffSteer(x0=x, W=3, R=1 / np.pi)

        nt.assert_array_almost_equal(block.T_output(0, 0, x=x)[0], x)
        nt.assert_array_almost_equal(block.T_deriv(0, 0, x=x), [0, 0, 0])

        nt.assert_array_almost_equal(block.T_output(5, -5, x=x)[0], x)
        nt.assert_array_almost_equal(block.T_deriv(5, -5, x=x), [0, 0, -10])

    @unittest.skip("cant test bdsim plot blocks")
    def test_vehicleplot(self):

        bike = Bicycle()
        block = VehiclePlot()

        s = State()
        block.T_start(s)
        block.T_step(np.array([0, 0, 0]))


class MultirotorBlockTest(unittest.TestCase):
    def test_multirotor(self):

        x = np.r_[[1, 2, 3, 0, 0, 0], np.zeros((6,))]
        block = MultiRotor(model=quadrotor)

        out = block.T_output(
            np.r_[614.675223, -614.675223, 614.675223, -614.675223], t=0, x=x
        )[0]
        self.assertIsInstance(out, dict)

        out = block.T_deriv(100*np.r_[1, 1, 1, 1], x=x)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (12,))

    def test_multirotormixer(self):

        block = MultiRotorMixer(model=quadrotor)
        nt.assert_array_almost_equal(
            block.T_output(0, 0, 0, -20, t=0)[0],
            [614.675223, -614.675223, 614.675223, -614.675223],
        )

    @unittest.skip("cant test bdsim plot blocks")
    def test_multirotorplot(self):

        block = MultiRotorPlot(model=quadrotor)

        class State:
            pass

        s = State()

        block.start(state=s)
        block.step(state=s)

    def test_quadrotor(self):

        block = MultiRotor(quadrotor)
        print(block.D)
        z = np.r_[0, 0, 0, 0]
        block.test_inputs = [z]
        nt.assert_equal(block.getstate0(), np.zeros((12,)))
        block.setstate(block.getstate0())

        x = block.getstate0()
        x[2] = -100  # set altitude
        u = 100 * np.r_[1, -1, 1, -1]

        # check outputs
        out = block.T_output(u, x=x)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

        self.assertIsInstance(out[0], dict)

        # check deriv, checked against MATLAB version 20200621
        u = 800 * np.r_[1, -1, 1, -1]  # too little thrust, falling
        d = block.T_deriv(u, x=x)
        self.assertIsInstance(d, np.ndarray)
        self.assertEqual(d.shape, (12,))
        self.assertGreater(d[8], 0)
        nt.assert_array_almost_equal(
            np.delete(d, 8), np.zeros((11,))
        )  # other derivs are zero

        u = 900 * np.r_[1, -1, 1, -1]  # too much thrust, rising
        self.assertLess(block.T_deriv(u, x=x)[8], 0)

        u = 800 * np.r_[1.2, -1, 0.8, -1]  # + pitch
        self.assertGreater(block.T_deriv(u, x=x)[10], 20)

        u = 800 * np.r_[0.8, -1, 1.2, -1]  # - pitch
        self.assertLess(block.T_deriv(u, x=x)[10], -20)

        u = 800 * np.r_[1, -0.8, 1, -1.2]  # + roll
        self.assertGreater(block.T_deriv(u, x=x)[9], 20)

        u = 800 * np.r_[1, -1.2, 1, -0.8]  # - roll
        self.assertLess(block.T_deriv(u, x=x)[9], -20)

    @unittest.skip("cant test bdsim plot blocks")
    def test_quadrotorplot(self):

        block = MultiRotor(quadrotor)
        u = [100 * np.r_[1, -1, 1, -1]]
        x = block.getstate0()
        out = block.T_output(u, x=x)[0]

        # block = MultiRotorPlot(quadrotor)
        # s = block.T_start()
        # block.T_step(out, s)


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    unittest.main()
