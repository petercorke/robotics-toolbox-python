import roboticstoolbox as rp
import unittest


class TestDHRobotDisplay(unittest.TestCase):
    def test_str_prismatic_precision(self):
        for link_type in (rp.PrismaticDH, rp.PrismaticMDH):
            robot = rp.DHRobot([link_type(qlim=[0.30479999999999996, 1.27])])
            output = str(robot)
            self.assertIn("0.3048", output)
            self.assertNotIn("0.30479999999999996", output)


if __name__ == "__main__":
    unittest.main()
