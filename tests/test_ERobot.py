#!/usr/bin/env python3
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import ERobot, ET, ETS, ERobot2, Link
from spatialmath import SE2, SE3
import unittest
import spatialmath as sm
import spatialgeometry as gm
from math import pi, sin, cos


class TestERobot(unittest.TestCase):
    # def test_init(self):
    #     ets = ETS(rtb.ET.Rz())
    #     robot = ERobot(
    #         ets, name="myname", manufacturer="I made it", comment="other stuff"
    #     )
    #     self.assertEqual(robot.name, "myname")
    #     self.assertEqual(robot.manufacturer, "I made it")
    #     self.assertEqual(robot.comment, "other stuff")

    # def test_init_ets(self):
    #     ets = (
    #         rtb.ET.tx(-0.0825)
    #         * rtb.ET.Rz()
    #         * rtb.ET.tx(-0.0825)
    #         * rtb.ET.tz()
    #         * rtb.ET.tx(0.1)
    #     )

    #     robot = ERobot(ets)
    #     self.assertEqual(robot.n, 2)
    #     self.assertIsInstance(robot[0], Link)
    #     self.assertIsInstance(robot[1], Link)
    #     self.assertTrue(robot[0].isrevolute)
    #     self.assertTrue(robot[1].isprismatic)

    #     self.assertIs(robot[0].parent, None)
    #     self.assertIs(robot[1].parent, robot[0])
    #     self.assertIs(robot[2].parent, robot[1])

    #     self.assertEqual(robot[0].children, [robot[1]])
    #     self.assertEqual(robot[1].children, [robot[2]])
    #     self.assertEqual(robot[2].children, [])

    # def test_init_elink(self):
    #     link1 = Link(ETS(ET.Rx()), name="link1")
    #     link2 = Link(ET.tx(1) * ET.ty(-0.5) * ET.tz(), name="link2", parent=link1)
    #     link3 = Link(ETS(ET.tx(1)), name="ee_1", parent=link2)
    #     robot = ERobot([link1, link2, link3])
    #     self.assertEqual(robot.n, 2)
    #     self.assertIsInstance(robot[0], Link)
    #     self.assertIsInstance(robot[1], Link)
    #     self.assertIsInstance(robot[2], Link)
    #     self.assertTrue(robot[0].isrevolute)
    #     self.assertTrue(robot[1].isprismatic)

    #     self.assertFalse(robot[2].isrevolute)
    #     self.assertFalse(robot[2].isprismatic)

    #     self.assertIs(robot[0].parent, None)
    #     self.assertIs(robot[1].parent, robot[0])
    #     self.assertIs(robot[2].parent, robot[1])

    #     self.assertEqual(robot[0].children, [robot[1]])
    #     self.assertEqual(robot[1].children, [robot[2]])
    #     self.assertEqual(robot[2].children, [])

    #     link1 = Link(ETS(ET.Rx()), name="link1")
    #     link2 = Link(ET.tx(1) * ET.ty(-0.5) * ET.tz(), name="link2", parent="link1")
    #     link3 = Link(ETS(ET.tx(1)), name="ee_1", parent="link2")
    #     robot = ERobot([link1, link2, link3])
    #     self.assertEqual(robot.n, 2)
    #     self.assertIsInstance(robot[0], Link)
    #     self.assertIsInstance(robot[1], Link)
    #     self.assertIsInstance(robot[2], Link)
    #     self.assertTrue(robot[0].isrevolute)
    #     self.assertTrue(robot[1].isprismatic)

    #     self.assertIs(robot[0].parent, None)
    #     self.assertIs(robot[1].parent, robot[0])
    #     self.assertIs(robot[2].parent, robot[1])

    #     self.assertEqual(robot[0].children, [robot[1]])
    #     self.assertEqual(robot[1].children, [robot[2]])
    #     self.assertEqual(robot[2].children, [])

    # def test_init_elink_autoparent(self):
    #     links = [
    #         Link(ETS(ET.Rx()), name="link1"),
    #         Link(ET.tx(1) * ET.ty(-0.5) * ET.tz(), name="link2"),
    #         Link(ETS(ET.tx(1)), name="ee_1"),
    #     ]
    #     robot = ERobot(links)
    #     self.assertEqual(robot.n, 2)
    #     self.assertIsInstance(robot[0], Link)
    #     self.assertIsInstance(robot[1], Link)
    #     self.assertIsInstance(robot[2], Link)
    #     self.assertTrue(robot[0].isrevolute)
    #     self.assertTrue(robot[1].isprismatic)
    #     self.assertIs(robot[0].parent, None)
    #     self.assertIs(robot[1].parent, robot[0])
    #     self.assertIs(robot[2].parent, robot[1])

    #     self.assertEqual(robot[0].children, [robot[1]])
    #     self.assertEqual(robot[1].children, [robot[2]])
    #     self.assertEqual(robot[2].children, [])

    # def test_init_elink_branched(self):
    #     robot = ERobot(
    #         [
    #             Link(ETS(ET.Rz()), name="link1"),
    #             Link(
    #                 ETS(ET.tx(1)) * ET.ty(-0.5) * ET.Rz(), name="link2", parent="link1"
    #             ),
    #             Link(ETS(ET.tx(1)), name="ee_1", parent="link2"),
    #             Link(ET.tx(1) * ET.ty(0.5) * ET.Rz(), name="link3", parent="link1"),
    #             Link(ETS(ET.tx(1)), name="ee_2", parent="link3"),
    #         ]
    #     )
    #     self.assertEqual(robot.n, 3)
    #     for i in range(5):
    #         self.assertIsInstance(robot[i], Link)
    #     self.assertTrue(robot[0].isrevolute)
    #     self.assertTrue(robot[1].isrevolute)
    #     self.assertTrue(robot[3].isrevolute)

    #     self.assertIs(robot[0].parent, None)
    #     self.assertIs(robot[1].parent, robot[0])
    #     self.assertIs(robot[2].parent, robot[1])
    #     self.assertIs(robot[3].parent, robot[0])
    #     self.assertIs(robot[4].parent, robot[3])

    #     self.assertEqual(robot[0].children, [robot[1], robot[3]])
    #     self.assertEqual(robot[1].children, [robot[2]])
    #     self.assertEqual(robot[2].children, [])
    #     self.assertEqual(robot[3].children, [robot[4]])
    #     self.assertEqual(robot[2].children, [])

    # def test_init_bases(self):
    #     e1 = Link()
    #     e2 = Link()
    #     e3 = Link(parent=e1)
    #     e4 = Link(parent=e2)

    #     with self.assertRaises(ValueError):
    #         ERobot([e1, e2, e3, e4])

    # def test_jindex(self):
    #     e1 = Link(ETS(ET.Rz()), jindex=0)
    #     e2 = Link(ETS(ET.Rz()), jindex=1, parent=e1)
    #     e3 = Link(ETS(ET.Rz()), jindex=2, parent=e2)
    #     e4 = Link(ETS(ET.Rz()), jindex=0, parent=e3)

    #     # with self.assertRaises(ValueError):
    #     ERobot([e1, e2, e3, e4], gripper_links=e4)

    # def test_jindex_fail(self):
    #     e1 = Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
    #     e2 = Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
    #     e3 = Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
    #     e4 = Link(rtb.ETS(rtb.ET.Rz()), jindex=5, parent=e3)

    #     with self.assertRaises(ValueError):
    #         ERobot([e1, e2, e3, e4])

    #     e1 = Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
    #     e2 = Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
    #     e3 = Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
    #     e4 = Link(rtb.ETS(rtb.ET.Rz()), parent=e3)

    #     with self.assertRaises(ValueError):
    #         ERobot([e1, e2, e3, e4])

    # def test_panda(self):
    #     panda = rtb.models.ETS.Panda()
    #     qz = np.array([0, 0, 0, 0, 0, 0, 0])
    #     qr = panda.qr

    #     nt.assert_array_almost_equal(panda.qr, qr)
    #     nt.assert_array_almost_equal(panda.qz, qz)
    #     nt.assert_array_almost_equal(panda.gravity, np.r_[0, 0, -9.81])

    # def test_q(self):
    #     panda = rtb.models.ETS.Panda()

    #     q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
    #     q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
    #     q3 = np.expand_dims(q1, 0)

    #     panda.q = q1
    #     nt.assert_array_almost_equal(panda.q, q1)
    #     panda.q = q2
    #     nt.assert_array_almost_equal(panda.q, q2)
    #     panda.q = q3
    #     nt.assert_array_almost_equal(np.expand_dims(panda.q, 0), q3)

    # def test_getters(self):
    #     panda = rtb.models.ETS.Panda()

    #     panda.qdd = np.ones((7, 1))
    #     panda.qd = np.ones((1, 7))
    #     panda.qdd = panda.qd
    #     panda.qd = panda.qdd

    # def test_control_mode(self):
    #     panda = rtb.models.ETS.Panda()
    #     panda.control_mode = "v"
    #     self.assertEqual(panda.control_mode, "v")

    # def test_base(self):
    #     panda = rtb.models.ETS.Panda()

    #     pose = sm.SE3()

    #     panda.base = pose.A
    #     nt.assert_array_almost_equal(np.eye(4), panda.base.A)

    #     panda.base = pose
    #     nt.assert_array_almost_equal(np.eye(4), panda.base.A)


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

    # def test_jacobe(self):
    #     pdh = rtb.models.DH.Panda()
    #     panda = rtb.models.ETS.Panda()
    #     q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
    #     panda.q = q1

    #     # nt.assert_array_almost_equal(panda.jacobe(), pdh.jacobe(q1))
    #     nt.assert_array_almost_equal(panda.jacobe(q1), pdh.jacobe(q1))

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

    def test_elinks(self):
        panda = rtb.models.Panda()
        self.assertEqual(panda.elinks[0], panda.link_dict[panda.elinks[0].name])

    def test_base_link_setter(self):
        panda = rtb.models.Panda()

        with self.assertRaises(TypeError):
            panda.base_link = [1]

    def test_ee_link_setter(self):
        panda = rtb.models.Panda()

        panda.ee_links = panda.links[5]

        with self.assertRaises(TypeError):
            panda.ee_links = [1]  # type: ignore




    # def test_control_mode2(self):
    #     panda = rtb.models.ETS.Panda()

    #     panda.control_mode = "p"

    #     with self.assertRaises(ValueError):
    #         panda.control_mode = "z"

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

        tau = robot.rne([0, -pi / 2], z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-1.5, 0])

        tau = robot.rne([-pi / 2, pi / 2], z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-0.5, -0.5])

        tau = robot.rne([-pi / 2, 0], z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        # check velocity terms
        robot.gravity = [0, 0, 0]
        q = [0, -pi / 2]
        h = -0.5 * sin(q[1])

        tau = robot.rne(q, [0, 0], z)
        nt.assert_array_almost_equal(tau, np.r_[0, 0] * h)

        tau = robot.rne(q, [1, 0], z)
        nt.assert_array_almost_equal(tau, np.r_[0, -1] * h)

        tau = robot.rne(q, [0, 1], z)
        nt.assert_array_almost_equal(tau, np.r_[1, 0] * h)

        tau = robot.rne(q, [1, 1], z)
        nt.assert_array_almost_equal(tau, np.r_[3, -1] * h)

        # check inertial terms

        d11 = 1.5 + cos(q[1])
        d12 = 0.25 + 0.5 * cos(q[1])
        d21 = d12
        d22 = 0.25

        tau = robot.rne(q, z, [0, 0])
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(q, z, [1, 0])
        nt.assert_array_almost_equal(tau, np.r_[d11, d21])

        tau = robot.rne(q, z, [0, 1])
        nt.assert_array_almost_equal(tau, np.r_[d12, d22])

        tau = robot.rne(q, z, [1, 1])
        nt.assert_array_almost_equal(tau, np.r_[d11 + d12, d21 + d22])


class TestERobot2(unittest.TestCase):
    def test_plot(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(robot.qz, block=False, name=True)
        e.close()

    def test_teach(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.teach(block=False, name=True)
        e.close()

        e = robot.teach(robot.qz, block=False, name=True)
        e.close()

    def test_plot_with_vellipse(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(
            robot.qz, block=False, name=True, vellipse=True, limits=[1, 2, 1, 2]
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


if __name__ == "__main__":  # pragma nocover

    unittest.main()
