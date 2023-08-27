"""
@author: Jesse Haviland
"""

import roboticstoolbox as rtb
from spatialmath import SE3
import numpy.testing as nt
import numpy as np
import unittest


class TestGripper(unittest.TestCase):
    def test_jindex_1(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
        e5 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=0, parent=e3)
        e7 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e5)

        rtb.Robot([e1, e2, e3, e5, e7], gripper_links=e5)

    def test_jindex_2(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
        e4 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e3)

        with self.assertRaises(ValueError):
            rtb.Robot([e1, e2, e3, e4], gripper_links=e4)

    def test_jindex_3(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=0)
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=1, parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=2, parent=e2)
        e4 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=0, parent=e3)
        e5 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e4)

        with self.assertRaises(ValueError):
            rtb.Robot([e1, e2, e3, e4, e5], gripper_links=e4)

    def test_jindex_4(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()))
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)
        e4 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e3)
        e5 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e4)
        e6 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e5)
        e7 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e6)

        rtb.Robot([e1, e2, e3, e4, e5, e6, e7], gripper_links=e3)

    def test_gripper_args(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()))
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)

        tool = SE3.Rx(1.0)

        g1 = rtb.Gripper([e2, e3], tool=tool)

        nt.assert_almost_equal(g1.tool.A, tool.A)

    def test_gripper_args2(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()))
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)

        tool = SE3.Rx(1.0)

        g2 = rtb.Gripper([e2, e3], tool=tool.A)

        nt.assert_almost_equal(g2.tool.A, tool.A)

    def test_init_fail(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()), jindex=1)
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1, jindex=2)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)

        with self.assertRaises(ValueError):
            rtb.Gripper([e2, e3])

    def test_str(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()))
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)

        tool = SE3.Rx(1.0)

        g1 = rtb.Gripper([e2, e3], tool=tool)

        s = g1.__str__()

        self.assertTrue(s.startswith('Gripper("", connected to , 2 joints, 2 links)'))

    def test_repr(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()))
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)

        g1 = rtb.Gripper([e2, e3])

        s = g1.__repr__()

        ans = """Gripper(['Link([ET.Rz(jindex=0)], name = "", parent="")', 'Link([ET.Rz(jindex=0)], name = "", parent="")'], name="", tool=None)"""  # noqa

        self.assertEqual(s, ans)

    def test_q(self):
        e1 = rtb.Link(rtb.ETS(rtb.ET.Rz()))
        e2 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e1)
        e3 = rtb.Link(rtb.ETS(rtb.ET.Rz()), parent=e2)

        g1 = rtb.Gripper([e2, e3])

        nt.assert_almost_equal(g1.q, np.zeros(2))
