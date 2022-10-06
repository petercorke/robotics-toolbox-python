"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import Link, ETS, ET, Robot
import spatialmath.base as sm
from spatialmath import SE3
import unittest
from copy import copy, deepcopy

from roboticstoolbox.robot.Robot import BaseRobot


class TestBaseRobot(unittest.TestCase):
    def test_init(self):

        links, name, urdf_string, urdf_filepath = rtb.Robot.URDF_read(
            "franka_description/robots/panda_arm_hand.urdf.xacro"
        )

        robot = rtb.Robot(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        robot.grippers[0].tool = SE3(0, 0, 0.1034)

    def test_init2(self):
        ets = rtb.ETS(rtb.ET.Rz())
        robot = rtb.Robot(
            ets, name="myname", manufacturer="I made it", comment="other stuff"
        )
        self.assertEqual(robot.name, "myname")
        self.assertEqual(robot.manufacturer, "I made it")
        self.assertEqual(robot.comment, "other stuff")

    def test_init3(self):
        l0 = Link()
        l1 = Link(parent=l0)
        r = Robot([l0, l1], base=SE3.Rx(1.3))
        # r.base_link = l1

        with self.assertRaises(TypeError):
            Robot(l0, base=SE3.Rx(1.3))  # type: ignore

        with self.assertRaises(TypeError):
            Robot([1, 2], base=SE3.Rx(1.3))  # type: ignore

    def test_init_ets(self):

        ets = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rz()
            * rtb.ET.tx(-0.0825)
            * rtb.ET.tz()
            * rtb.ET.tx(0.1)
        )

        robot = rtb.Robot(ets)

        self.assertEqual(robot.n, 2)
        self.assertIsInstance(robot.links[0], rtb.Link)
        self.assertIsInstance(robot.links[1], rtb.Link)
        self.assertTrue(robot.links[0].isrevolute)
        self.assertTrue(robot.links[1].isprismatic)

        self.assertIs(robot.links[0].parent, None)
        self.assertIs(robot.links[1].parent, robot.links[0])
        self.assertIs(robot.links[2].parent, robot.links[1])

        self.assertEqual(robot.links[0].children, [robot.links[1]])
        self.assertEqual(robot.links[1].children, [robot.links[2]])
        self.assertEqual(robot.links[2].children, [])

    def test_init_elink(self):
        link1 = Link(ETS(ET.Rx()), name="link1")
        link2 = Link(ET.tx(1) * ET.ty(-0.5) * ET.tz(), name="link2", parent=link1)
        link3 = Link(ETS(ET.tx(1)), name="ee_1", parent=link2)
        robot = Robot([link1, link2, link3])
        self.assertEqual(robot.n, 2)
        self.assertIsInstance(robot.links[0], Link)
        self.assertIsInstance(robot.links[1], Link)
        self.assertIsInstance(robot.links[2], Link)
        self.assertTrue(robot.links[0].isrevolute)
        self.assertTrue(robot.links[1].isprismatic)

        self.assertFalse(robot.links[2].isrevolute)
        self.assertFalse(robot.links[2].isprismatic)

        self.assertIs(robot.links[0].parent, None)
        self.assertIs(robot.links[1].parent, robot.links[0])
        self.assertIs(robot.links[2].parent, robot.links[1])

        self.assertEqual(robot.links[0].children, [robot.links[1]])
        self.assertEqual(robot.links[1].children, [robot.links[2]])
        self.assertEqual(robot.links[2].children, [])

        link1 = Link(ETS(ET.Rx()), name="link1")
        link2 = Link(ET.tx(1) * ET.ty(-0.5) * ET.tz(), name="link2", parent="link1")
        link3 = Link(ETS(ET.tx(1)), name="ee_1", parent="link2")
        robot = Robot([link1, link2, link3])
        self.assertEqual(robot.n, 2)
        self.assertIsInstance(robot.links[0], Link)
        self.assertIsInstance(robot.links[1], Link)
        self.assertIsInstance(robot.links[2], Link)
        self.assertTrue(robot.links[0].isrevolute)
        self.assertTrue(robot.links[1].isprismatic)

        self.assertIs(robot.links[0].parent, None)
        self.assertIs(robot.links[1].parent, robot.links[0])
        self.assertIs(robot.links[2].parent, robot.links[1])

        self.assertEqual(robot[0].children, [robot[1]])
        self.assertEqual(robot[1].children, [robot[2]])
        self.assertEqual(robot[2].children, [])

    def test_init_elink_autoparent(self):
        links = [
            Link(ETS(ET.Rx()), name="link1"),
            Link(ET.tx(1) * ET.ty(-0.5) * ET.tz(), name="link2"),
            Link(ETS(ET.tx(1)), name="ee_1"),
        ]
        robot = Robot(links)
        self.assertEqual(robot.n, 2)
        self.assertIsInstance(robot[0], Link)
        self.assertIsInstance(robot[1], Link)
        self.assertIsInstance(robot[2], Link)
        self.assertTrue(robot[0].isrevolute)
        self.assertTrue(robot[1].isprismatic)
        self.assertIs(robot[0].parent, None)
        self.assertIs(robot[1].parent, robot[0])
        self.assertIs(robot[2].parent, robot[1])

        self.assertEqual(robot[0].children, [robot[1]])
        self.assertEqual(robot[1].children, [robot[2]])
        self.assertEqual(robot[2].children, [])

    def test_init_elink_branched(self):
        robot = Robot(
            [
                Link(ETS(ET.Rz()), name="link1"),
                Link(
                    ETS(ET.tx(1)) * ET.ty(-0.5) * ET.Rz(), name="link2", parent="link1"
                ),
                Link(ETS(ET.tx(1)), name="ee_1", parent="link2"),
                Link(ET.tx(1) * ET.ty(0.5) * ET.Rz(), name="link3", parent="link1"),
                Link(ETS(ET.tx(1)), name="ee_2", parent="link3"),
            ]
        )
        self.assertEqual(robot.n, 3)
        for i in range(5):
            self.assertIsInstance(robot[i], Link)
        self.assertTrue(robot[0].isrevolute)
        self.assertTrue(robot[1].isrevolute)
        self.assertTrue(robot[3].isrevolute)

        self.assertIs(robot[0].parent, None)
        self.assertIs(robot[1].parent, robot[0])
        self.assertIs(robot[2].parent, robot[1])
        self.assertIs(robot[3].parent, robot[0])
        self.assertIs(robot[4].parent, robot[3])

        self.assertEqual(robot[0].children, [robot[1], robot[3]])
        self.assertEqual(robot[1].children, [robot[2]])
        self.assertEqual(robot[2].children, [])
        self.assertEqual(robot[3].children, [robot[4]])
        self.assertEqual(robot[2].children, [])

    def test_init_bases(self):
        e1 = Link()
        e2 = Link()
        e3 = Link(parent=e1)
        e4 = Link(parent=e2)

        with self.assertRaises(ValueError):
            Robot([e1, e2, e3, e4])

    def test_jindex(self):
        e1 = Link(ETS(ET.Rz()), jindex=0)
        e2 = Link(ETS(ET.Rz()), jindex=1, parent=e1)
        e3 = Link(ETS(ET.Rz()), jindex=2, parent=e2)
        e4 = Link(ETS(ET.Rz()), jindex=0, parent=e3)

        # with self.assertRaises(ValueError):
        Robot([e1, e2, e3, e4], gripper_links=e4)

    def test_jindex_fail(self):
        e1 = Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
        e2 = Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
        e3 = Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
        e4 = Link(rtb.ETS(rtb.ET.Rz()), jindex=5, parent=e3)

        with self.assertRaises(ValueError):
            Robot([e1, e2, e3, e4])

        e1 = Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
        e2 = Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
        e3 = Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
        e4 = Link(rtb.ETS(rtb.ET.Rz()), parent=e3)

        with self.assertRaises(ValueError):
            Robot([e1, e2, e3, e4])

    def test_panda(self):
        panda = rtb.models.ETS.Panda()
        qz = np.array([0, 0, 0, 0, 0, 0, 0])
        qr = panda.qr

        nt.assert_array_almost_equal(panda.qr, qr)
        nt.assert_array_almost_equal(panda.qz, qz)
        nt.assert_array_almost_equal(panda.gravity, np.array([0, 0, -9.81]))

    def test_q(self):
        panda = rtb.models.ETS.Panda()

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)

        panda.q = q1
        nt.assert_array_almost_equal(panda.q, q1)
        panda.q = q2
        nt.assert_array_almost_equal(panda.q, q2)
        panda.q = q3
        nt.assert_array_almost_equal(np.expand_dims(panda.q, 0), q3)

    def test_getters(self):
        panda = rtb.models.ETS.Panda()

        panda.qdd = np.ones((7, 1))
        panda.qd = np.ones((1, 7))
        panda.qdd = panda.qd
        panda.qd = panda.qdd

    def test_control_mode(self):
        panda = rtb.models.ETS.Panda()
        panda.control_mode = "v"
        self.assertEqual(panda.control_mode, "v")

    def test_base(self):
        panda = rtb.models.ETS.Panda()

        pose = SE3()

        panda.base = pose.A
        nt.assert_array_almost_equal(np.eye(4), panda.base.A)

        panda.base = pose
        nt.assert_array_almost_equal(np.eye(4), panda.base.A)

    def test_control_mode2(self):
        panda = rtb.models.ETS.Panda()

        panda.control_mode = "p"

        with self.assertRaises(ValueError):
            panda.control_mode = "z"

    def test_manuf(self):
        panda = rtb.models.ETS.Panda()

        self.assertIsInstance(panda.manufacturer, str)
