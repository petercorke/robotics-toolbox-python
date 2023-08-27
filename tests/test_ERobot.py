#!/usr/bin/env python3
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import ERobot, ET, ETS, Link

# from spatialmath import SE2, SE3
import unittest
import spatialmath as sm
import spatialgeometry as gm
from math import pi, sin, cos

try:
    from sympy import symbols

    _sympy = True
except ModuleNotFoundError:
    _sympy = False


class TestERobot(unittest.TestCase):
    def test_jacobm(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [1.27080875e-17],
                [2.38242538e-02],
                [6.61029519e-03],
                [8.18202121e-03],
                [7.74546204e-04],
                [-1.10885380e-02],
                [0.00000000e00],
            ]
        )

        panda.q = q1
        nt.assert_array_almost_equal(panda.jacobm(), ans)
        nt.assert_array_almost_equal(panda.jacobm(q2), ans)
        nt.assert_array_almost_equal(panda.jacobm(q3), ans)
        nt.assert_array_almost_equal(panda.jacobm(q4), ans)
        nt.assert_array_almost_equal(panda.jacobm(J=panda.jacob0(q1)), ans)
        # self.assertRaises(ValueError, panda.jacobm)
        self.assertRaises(TypeError, panda.jacobm, "Wfgsrth")
        self.assertRaises(ValueError, panda.jacobm, [1, 3], np.array([1, 5]))
        self.assertRaises(TypeError, panda.jacobm, [1, 3], "qwe")
        self.assertRaises(TypeError, panda.jacobm, [1, 3], panda.jacob0(q1), [1, 2, 3])
        self.assertRaises(
            ValueError, panda.jacobm, [1, 3], panda.jacob0(q1), np.array([1, 2, 3])
        )

    def test_dict(self):
        panda = rtb.models.Panda()
        panda.grippers[0].links[0].collision.append(gm.Cuboid([1, 1, 1]))
        panda._to_dict()

        wx = rtb.models.wx250s()
        wx._to_dict()

    def test_fkdict(self):
        panda = rtb.models.Panda()
        panda.grippers[0].links[0].collision.append(gm.Cuboid([1, 1, 1]))
        panda._fk_dict()

    def test_dist(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], pose=sm.SE3(3, 0, 0))
        p = rtb.models.Panda()

        d0, _, _ = p.closest_point(p.q, s0)
        d1, _, _ = p.closest_point(p.q, s1, 5)
        d2, _, _ = p.closest_point(p.q, s1)

        self.assertAlmostEqual(d0, -0.5599999999995913)  # type: ignore
        self.assertAlmostEqual(d1, 2.362147178773918)  # type: ignore
        self.assertAlmostEqual(d2, None)  # type: ignore

    def test_collided(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], pose=sm.SE3(3, 0, 0))
        p = rtb.models.Panda()

        c0 = p.collided(p.q, s0)
        c1 = p.collided(p.q, s1)

        self.assertTrue(c0)
        self.assertFalse(c1)

    def test_invdyn(self):
        # create a 2 link robot
        # Example from Spong etal. 2nd edition, p. 260
        l1 = Link(ets=ETS(ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
        l2 = Link(ets=ETS(ET.tx(1)) * ET.Ry(), m=1, r=[0.5, 0, 0], parent=l1, name="l2")
        robot = ERobot([l1, l2], name="simple 2 link")
        z = np.zeros(robot.n)

        # check gravity load
        tau = robot.rne(z, z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-2, -0.5])

        tau = robot.rne(np.array([0.0, -pi / 2.0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-1.5, 0])

        tau = robot.rne(np.array([-pi / 2, pi / 2]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-0.5, -0.5])

        tau = robot.rne(np.array([-pi / 2, 0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        # check velocity terms
        robot.gravity = [0, 0, 0]
        q = np.array([0, -pi / 2])
        h = -0.5 * sin(q[1])

        tau = robot.rne(q, np.array([0, 0]), z)
        nt.assert_array_almost_equal(tau, np.r_[0, 0] * h)

        tau = robot.rne(q, np.array([1, 0]), z)
        nt.assert_array_almost_equal(tau, np.r_[0, -1] * h)

        tau = robot.rne(q, np.array([0, 1]), z)
        nt.assert_array_almost_equal(tau, np.r_[1, 0] * h)

        tau = robot.rne(q, np.array([1, 1]), z)
        nt.assert_array_almost_equal(tau, np.r_[3, -1] * h)

        # check inertial terms

        d11 = 1.5 + cos(q[1])
        d12 = 0.25 + 0.5 * cos(q[1])
        d21 = d12
        d22 = 0.25

        tau = robot.rne(q, z, np.array([0, 0]))
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(q, z, np.array([1, 0]))
        nt.assert_array_almost_equal(tau, np.r_[d11, d21])

        tau = robot.rne(q, z, np.array([0, 1]))
        nt.assert_array_almost_equal(tau, np.r_[d12, d22])

        tau = robot.rne(q, z, np.array([1, 1]))
        nt.assert_array_almost_equal(tau, np.r_[d11 + d12, d21 + d22])


class TestERobot2(unittest.TestCase):
    def test_plot(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(robot.qz, block=False, name=True)
        e.close()

    def test_teach(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.teach(robot.qz, block=False)
        e.close()

        e = robot.teach(robot.qz, block=False)
        e.close()

    def test_plot_with_vellipse(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(
            robot.qb, block=False, name=True, vellipse=True, limits=[1, 2, 1, 2]
        )
        e.step()
        e.close()

    def test_plot_with_fellipse(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(
            robot.qz, block=False, name=True, dellipse=True, limits=[1, 2, 1, 2]
        )
        e.step()
        e.close()

    def test_base(self):
        robot = rtb.models.ETS.Planar2()
        nt.assert_almost_equal(robot.base.A, sm.SE2().A)

    def test_jacobe(self):
        robot = rtb.models.ETS.Planar2()
        J = robot.jacobe(robot.qz)

        a1 = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 1.0]])

        nt.assert_almost_equal(J, a1)

    @unittest.skipUnless(_sympy, "sympy not installed")
    def test_symdyn(self):

        a1, a2, r1, r2, m1, m2, g = symbols("a1 a2 r1 r2 m1 m2 g")
        link1 = Link(ET.Ry(flip=True), m=m1, r=[r1, 0, 0], name="link0")
        link2 = Link(ET.tx(a1) * ET.Ry(flip=True), m=m2, r=[r2, 0, 0], name="link1")
        robot = ERobot([link1, link2])

        q = symbols("q:2")
        qd = symbols("qd:2")
        qdd = symbols("qdd:2")
        Q = robot.rne(q, qd, qdd, gravity=[0, 0, g], symbolic=True)

        self.assertEqual(
            str(Q[0]),
            "a1**2*m2*qd0**2*sin(q1)*cos(q1) + a1*qd0*(-a1*m2*qd0*cos(q1) - m2*r2*(qd0 + qd1))*sin(q1) - a1*(m2*(a1*qd0*qd1*cos(q1) - a1*qdd0*sin(q1) - g*sin(q0)*cos(q1) - g*sin(q1)*cos(q0)) + (qd0 + qd1)*(-a1*m2*qd0*cos(q1) - m2*r2*(qd0 + qd1)))*sin(q1) - a1*(-a1*m2*qd0*(-qd0 - qd1)*sin(q1) - m2*r2*(qdd0 + qdd1) + m2*(-a1*qd0*qd1*sin(q1) - a1*qdd0*cos(q1) + g*sin(q0)*sin(q1) - g*cos(q0)*cos(q1)))*cos(q1) + g*m1*r1*cos(q0) + m1*qdd0*r1**2 + m2*r2**2*(qdd0 + qdd1) - m2*r2*(-a1*qd0*qd1*sin(q1) - a1*qdd0*cos(q1) + g*sin(q0)*sin(q1) - g*cos(q0)*cos(q1))",
        )
        self.assertEqual(
            str(Q[1]),
            "a1**2*m2*qd0**2*sin(q1)*cos(q1) + a1*qd0*(-a1*m2*qd0*cos(q1) - m2*r2*(qd0 + qd1))*sin(q1) + m2*r2**2*(qdd0 + qdd1) - m2*r2*(-a1*qd0*qd1*sin(q1) - a1*qdd0*cos(q1) + g*sin(q0)*sin(q1) - g*cos(q0)*cos(q1))",
        )


if __name__ == "__main__":  # pragma nocover

    unittest.main()
