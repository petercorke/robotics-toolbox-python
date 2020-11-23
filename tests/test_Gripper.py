"""
@author: Jesse Haviland
"""

import roboticstoolbox as rtb
import unittest


class TestGripper(unittest.TestCase):

    def test_jindex_1(self):
        e1 = rtb.ELink(rtb.ETS.rz(), jindex=0)
        e2 = rtb.ELink(rtb.ETS.rz(), jindex=1, parent=e1)
        e3 = rtb.ELink(rtb.ETS.rz(), jindex=2, parent=e2)
        e5 = rtb.ELink(rtb.ETS.rz(), jindex=0, parent=e3)
        e7 = rtb.ELink(rtb.ETS.rz(), jindex=1, parent=e5)

        rtb.ERobot([e1, e2, e3, e5, e7], gripper_links=e5)

    def test_jindex_2(self):
        e1 = rtb.ELink(rtb.ETS.rz(), jindex=0)
        e2 = rtb.ELink(rtb.ETS.rz(), jindex=1, parent=e1)
        e3 = rtb.ELink(rtb.ETS.rz(), jindex=2, parent=e2)
        e4 = rtb.ELink(rtb.ETS.rz(), jindex=1, parent=e3)

        with self.assertRaises(ValueError):
            rtb.ERobot([e1, e2, e3, e4], gripper_links=e4)

    def test_jindex_3(self):
        e1 = rtb.ELink(rtb.ETS.rz(), jindex=0)
        e2 = rtb.ELink(rtb.ETS.rz(), jindex=1, parent=e1)
        e3 = rtb.ELink(rtb.ETS.rz(), jindex=2, parent=e2)
        e4 = rtb.ELink(rtb.ETS.rz(), jindex=0, parent=e3)
        e5 = rtb.ELink(rtb.ETS.rz(), parent=e4)

        with self.assertRaises(ValueError):
            rtb.ERobot([e1, e2, e3, e4, e5], gripper_links=e4)

    def test_jindex_4(self):
        e1 = rtb.ELink(rtb.ETS.rz())
        e2 = rtb.ELink(rtb.ETS.rz(), parent=e1)
        e3 = rtb.ELink(rtb.ETS.rz(), parent=e2)
        e4 = rtb.ELink(rtb.ETS.rz(), parent=e3)
        e5 = rtb.ELink(rtb.ETS.rz(), parent=e4)
        e6 = rtb.ELink(rtb.ETS.rz(), parent=e5)
        e7 = rtb.ELink(rtb.ETS.rz(), parent=e6)

        rtb.ERobot([e1, e2, e3, e4, e5, e6, e7], gripper_links=e3)
