"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import Link, ETS, ET, Robot
from spatialmath import SE3
import unittest
from copy import deepcopy
from swift import Swift

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
        # l1 = Link(parent=l0)
        # r = Robot([l0, l1], base=SE3.Rx(1.3))
        # r.base_link = l1

        with self.assertRaises(TypeError):
            Robot(l0, base=SE3.Rx(1.3))  # type: ignore

        with self.assertRaises(TypeError):
            Robot([1, 2], base=SE3.Rx(1.3))  # type: ignore

    def test_init4(self):
        ets = ETS(rtb.ET.Rz())
        robot = Robot(
            ets, name="myname", manufacturer="I made it", comment="other stuff"
        )
        self.assertEqual(robot.name, "myname")
        self.assertEqual(robot.manufacturer, "I made it")
        self.assertEqual(robot.comment, "other stuff")

    def test_init5(self):

        base = SE3.Trans(0, 0, 0.1).A
        ets = ETS(rtb.ET.Rz())
        robot = Robot(ets, base=base, tool=base)
        nt.assert_almost_equal(robot.base.A, base)
        nt.assert_almost_equal(robot.tool.A, base)

    def test_init6(self):

        base = SE3.Trans(0, 0, 0.1)
        ets = ETS(rtb.ET.Rz())
        robot = Robot(ets, base=base, tool=base)
        nt.assert_almost_equal(robot.base.A, base.A)
        nt.assert_almost_equal(robot.tool.A, base.A)

    def test_init7(self):

        keywords = 2
        ets = ETS(rtb.ET.Rz())

        with self.assertRaises(TypeError):
            Robot(ets, keywords=keywords)  # type: ignore

    def test_init8(self):

        links = [2, 3, 4, 5]

        with self.assertRaises(TypeError):
            BaseRobot(links=links)  # type: ignore

    def test_init9(self):

        robot = rtb.models.Panda()
        robot2 = rtb.models.PR2()

        self.assertTrue(robot2._hasdynamics)
        self.assertTrue(robot._hasgeometry)
        self.assertTrue(robot._hascollision)

    def test_init10(self):

        links = [Link(name="link1"), Link(name="link1"), Link(name="link1")]

        with self.assertRaises(ValueError):
            Robot(links)

    def test_init11(self):

        l1 = Link(parent="l3")
        l2 = Link(parent=l1)
        l3 = Link(parent=l2, name="l3")

        links = [l1, l2, l3]

        with self.assertRaises(ValueError):
            Robot(links)

    def test_init12(self):

        l1 = Link(jindex=1, ets=rtb.ET.Rz())
        l2 = Link(jindex=2, parent=l1, ets=rtb.ET.Rz())
        l3 = Link(parent=l2, ets=rtb.ET.Rz())

        links = [l1, l2, l3]

        with self.assertRaises(ValueError):
            Robot(links)

    def test_iter(self):
        robot = rtb.models.Panda()
        for link in robot:
            self.assertIsInstance(link, Link)

    def test_get(self):
        panda = rtb.models.ETS.Panda()
        self.assertIsInstance(panda[1], Link)
        self.assertIsInstance(panda["link0"], Link)

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

    def test_str(self):
        panda = rtb.models.Panda()
        pr2 = rtb.models.PR2()
        self.assertIsInstance(str(panda), str)
        self.assertIsInstance(str(pr2), str)
        self.assertIsInstance(repr(panda), str)
        self.assertIsInstance(repr(pr2), str)

    def test_nlinks(self):
        panda = rtb.models.Panda()
        self.assertEqual(panda.nlinks, 12)

    def test_configs(self):
        panda = rtb.models.Panda()
        configs = panda.configs

        nt.assert_equal(configs["qr"], panda.qr)
        nt.assert_equal(configs["qz"], panda.qz)

    def test_keywords(self):
        panda = Robot(
            ETS([ET.Rz(qlim=[-1, 1]), ET.tz(qlim=[-1, 1]), ET.SE3(SE3.Trans(1, 2, 3))]),
            keywords=["test"],
        )
        self.assertEqual(panda.keywords, ["test"])
        self.assertFalse(panda.symbolic)
        self.assertFalse(panda.hasdynamics)
        self.assertFalse(panda.hasgeometry)
        self.assertFalse(panda.hascollision)
        self.assertEqual(panda.default_backend, None)
        panda.default_backend = "Swift"

        self.assertEqual(panda.qlim[0, 0], -1.0)

    def test_qlim(self):
        panda = Robot(ETS([ET.Rz(qlim=[-1, 1]), ET.tz()]), keywords=["test"])

        with self.assertRaises(ValueError):
            panda.qlim

    def test_joint_types(self):
        panda = Robot(
            ETS([ET.Rz(qlim=[-1, 1]), ET.tz(qlim=[-1, 1]), ET.SE3(SE3.Trans(1, 2, 3))]),
        )

        self.assertTrue(panda.prismaticjoints[1])
        self.assertTrue(panda.revolutejoints[0])

    def test_urdf_string(self):
        panda = rtb.models.Panda()
        self.assertIsInstance(panda.urdf_string, str)
        self.assertIsInstance(panda.urdf_filepath, str)

    def test_manipulability_fail(self):
        puma = rtb.models.Puma560()
        puma.q = puma.qr

        with self.assertRaises(ValueError):
            puma.manipulability(method="notamethod")  # type: ignore

    def test_tool(self):
        panda = rtb.models.Panda()

        panda.tool = SE3.Ry(0.5)
        nt.assert_almost_equal(panda.tool.A, SE3.Ry(0.5).A)

        panda.tool = SE3.Ry(0.5).A
        nt.assert_almost_equal(panda.tool.A, SE3.Ry(0.5).A)

    def test_get_path(self):
        panda = rtb.models.Panda()

        links, n, tool = panda.get_path(end="panda_link1", start="panda_link4")

        for link in links:
            print(link.name)

        self.assertEqual(links[0].name, "panda_link4")
        self.assertEqual(links[1].name, "panda_link3")
        self.assertEqual(links[2].name, "panda_link2")
        self.assertEqual(links[3].name, "panda_link1")

        self.assertEqual(n, 4)
        nt.assert_equal(tool, np.eye(4))

    def test_get_path2(self):
        panda = rtb.models.Panda()

        links, n, tool = panda.get_path(end="panda_link4", start="panda_link1")

        for link in links:
            print(link.name)

        self.assertEqual(links[0].name, "panda_link1")
        self.assertEqual(links[1].name, "panda_link2")
        self.assertEqual(links[2].name, "panda_link3")
        self.assertEqual(links[3].name, "panda_link4")

        self.assertEqual(n, 4)
        nt.assert_equal(tool, np.eye(4))

    def test_get_path_fail(self):
        panda = rtb.models.Panda()

        with self.assertRaises(ValueError):
            _, _, _ = panda.get_path(end="panda", start="panda")

    def test_getlink(self):
        panda = rtb.models.Panda()

        l0 = panda._getlink(panda.grippers[0])

        self.assertEqual(l0.name, "panda_hand")

    def test_getlink_fail(self):
        panda = rtb.models.Panda()

        with self.assertRaises(ValueError):
            panda._getlink(rtb.Link(rtb.ETS()))

    def test_getlink_fail2(self):
        panda = rtb.models.Panda()

        with self.assertRaises(ValueError):
            panda._getlink(rtb.Gripper([rtb.Link(rtb.ETS())]))

    def test_getlink_fail3(self):
        panda = rtb.models.Panda()

        with self.assertRaises(TypeError):
            panda._getlink(2.0)

    def test_limits(self):
        r = rtb.models.YuMi()

        end, _, _ = r._get_limit_links()

        self.assertEqual(end.name, "r_gripper")

    def test_limits2(self):
        r = rtb.models.Panda()

        end, _, _ = r._get_limit_links()

        self.assertEqual(end.name, "panda_hand")

    def test_limits3(self):
        l1 = rtb.Link(rtb.ETS(), name="l1")
        l2 = rtb.Link(rtb.ETS(), parent=l1, name="l2")
        l3 = rtb.Link(rtb.ETS(), parent=l1, name="l3")

        r = rtb.Robot([l1, l2, l3])

        end, _, _ = r._get_limit_links()

        self.assertEqual(end.name, "l2")

    def test_limits4(self):
        r = rtb.models.Panda()

        end, _, _ = r._get_limit_links(end=r.grippers[0])

        self.assertEqual(end.name, "panda_hand")

    def test_ets_gripper(self):
        r = rtb.models.Panda()

        ets = r.ets(start=r.grippers[0])

        self.assertEqual(ets.n, 0)

    def test_ets_gripper2(self):
        r = rtb.models.YuMi()

        ets = r.ets()

        self.assertEqual(ets.n, 7)

    def test_ets_gripper3(self):
        l1 = rtb.Link(rtb.ETS(), name="l1")
        l2 = rtb.Link(rtb.ETS(), parent=l1, name="l2")
        l3 = rtb.Link(rtb.ETS(), parent=l1, name="l3")

        r = rtb.Robot([l1, l2, l3])

        ets = r.ets()

        self.assertEqual(ets.n, 0)

    def test_ets_gripper4(self):
        r = rtb.models.Panda()

        ets = r.ets(end=r.grippers[0])

        self.assertEqual(ets.n, 7)

    def test_ets(self):
        r = rtb.models.Panda()

        ets = r.ets(end=r.links[4], start=r.links[4])

        self.assertEqual(ets.n, 1)

    def test_copy(self):
        r = rtb.models.Panda()

        r2 = deepcopy(r)

        self.assertTrue(r2.links[2] != r.links[2])

    def test_copy2(self):
        r = rtb.models.DH.Panda()

        r2 = deepcopy(r)

        self.assertTrue(r2.links[2] != r.links[2])

    def test_copy3(self):
        r = rtb.Robot2(rtb.ETS2(rtb.ET2.R()))

        r2 = deepcopy(r)

        self.assertTrue(r2.links[0] != r.links[0])

    def test_toradians(self):
        r = rtb.models.Panda()

        q = np.ones((2, 7))

        qn = r.toradians(q)

        nt.assert_equal(qn, q * np.pi / 180.0)

    def test_todegrees(self):
        r = rtb.models.Panda()

        q = np.ones((2, 7))

        qn = r.todegrees(q)

        nt.assert_equal(qn, q * 180.0 / np.pi)

    def test_random_q(self):
        r = rtb.models.Panda()

        q = r.random_q()

        self.assertEqual(q.shape[0], 7)

    def test_heirarchy(self):
        r = rtb.models.Panda()

        r.hierarchy()

    def test_segments2(self):
        r = rtb.models.YuMi()

        segs = r.segments()

        self.assertEqual(len(segs), 7)

    def test_get_backend(self):
        r = rtb.models.Panda()

        be = r._get_graphical_backend()

        self.assertTrue(isinstance(be, Swift))

    def test_teach(self):
        robot = rtb.models.ETS.Panda()
        e = robot.teach(q=None, block=False, vellipse=True, fellipse=True)
        e.close()
