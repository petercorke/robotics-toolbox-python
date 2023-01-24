from collections import namedtuple
from math import pi
import numpy.testing as nt
import numpy as np
import unittest
import spatialmath.base as sm

from roboticstoolbox.mobile import *


class TestPlanners(unittest.TestCase):

    def test_occgrid(self):
        g = np.zeros((100, 120))
        g[20:30, 50:80] = 1

        og = BinaryOccupancyGrid(g, name='my grid')

        self.assertEqual(og.shape, g.shape)
        
        s = str(og)
        self.assertIsInstance(s, str)
        self.assertEqual(s, "BinaryOccupancyGrid[my grid]: 120 x 100, cell size=1, x = [0.0, 119.0], y = [0.0, 99.0], 2.5% occupied")

        self.assertEqual(og.xmin, 0)
        self.assertEqual(og.xmax, 119)
        self.assertEqual(og.ymin, 0)
        self.assertEqual(og.ymax, 99)

        self.assertTrue(og.isoccupied((50, 20)))
        self.assertTrue(og.isoccupied((60, 25)))
        self.assertTrue(og.isoccupied((200, 200)))
        self.assertFalse(og.isoccupied((0, 0)))
        self.assertFalse(og.isoccupied((80, 30)))

        og.plot(block=False)

        og2 = og.copy()

        self.assertEqual(og2.xmin, 0)
        self.assertEqual(og2.xmax, 119)
        self.assertEqual(og2.ymin, 0)
        self.assertEqual(og2.ymax, 99)

        self.assertTrue(og2.isoccupied((50, 20)))
        self.assertTrue(og2.isoccupied((60, 25)))
        self.assertTrue(og2.isoccupied((200, 200)))
        self.assertFalse(og2.isoccupied((0, 0)))
        self.assertFalse(og2.isoccupied((80, 30)))
        self.assertFalse(og2.isoccupied((45, 20)))

        og2.inflate(5)

        self.assertTrue(og2.isoccupied((50, 20)))
        self.assertTrue(og2.isoccupied((60, 25)))
        self.assertTrue(og2.isoccupied((200, 200)))
        self.assertFalse(og2.isoccupied((0, 0)))
        self.assertTrue(og2.isoccupied((80, 30)))
        self.assertTrue(og2.isoccupied((45, 20)))

        self.assertEqual(str(og2), "BinaryOccupancyGrid[my grid]: 120 x 100, cell size=1, x = [0.0, 119.0], y = [0.0, 99.0], 6.3% occupied")

        # check no change to original
        self.assertFalse(og.isoccupied((80, 30)))

        og = BinaryOccupancyGrid(g, cellsize=0.1, origin=(2,4), name='foo')

        self.assertEqual(og.xmin, 2)
        self.assertEqual(og.xmax, 13.9)
        self.assertEqual(og.ymin, 4)
        self.assertEqual(og.ymax, 13.9)
        self.assertTrue(og.isoccupied((8.5, 6.5)))
        self.assertTrue(og.isoccupied((500, 500)))
        self.assertFalse(og.isoccupied((3, 5)))

        og.inflate(0.5)
        self.assertEqual(str(og), "BinaryOccupancyGrid[foo]: 120 x 100, cell size=0.1, x = [2.0, 13.9], y = [4.0, 13.9], 6.3% occupied")

    def test_bug2(self):
        pass

    def test_dubins(self):

        start = (0, 0, pi/2)
        goal = (1, 0, pi/2)

        dubins = DubinsPlanner(curvature=1.0)
        path, status = dubins.query(start, goal)

        self.assertIsInstance(path, np.ndarray)
        self.assertEqual(path.shape, (74,3))
        self.assertEqual(status.__class__.__name__, "DubinsStatus")
        self.assertTrue(hasattr(status, 'segments'))
        self.assertTrue(hasattr(status, 'length'))

    def test_reedsshepp(self):

        start = (0, 0, pi/2)
        goal = (1, 0, pi/2)

        rs = ReedsSheppPlanner(curvature=1.0)
        path, status = rs.query(start, goal)

        self.assertIsInstance(path, np.ndarray)
        self.assertEqual(path.shape, (65,3))
        self.assertEqual(status.__class__.__name__, "ReedsSheppStatus")
        self.assertTrue(hasattr(status, 'segments'))
        self.assertTrue(hasattr(status, 'length'))
        self.assertTrue(hasattr(status, 'direction'))

    # def test_bug2(self):

    #     vars = loadmat("data/map1.mat")
    #     map = vars['map']

    #     bug = Bug2Planner(map)
    #     # bug.plan()
    #     path = bug.query([20, 10], [50, 35])

    #     # valid path
    #     self.assertTrue(path is not None)

    #     # valid Nx2 array
    #     self.assertIsInstance(path, np.ndarray)
    #     self.assertEqual(path.shape[1], 2)

    #     # includes start and goal
    #     self.assertTrue(all(path[0,:] == [20,10]))
    #     self.assertTrue(all(path[-1,:] == [50,35]))

    #     # path doesn't include obstacles
    #     for p in path:
    #         self.assertFalse(bug.is_occupied(p))

    #     # there are no gaps
    #     for k in range(len(path)-1):
    #         d = np.linalg.norm(path[k] - path[k+1])
    #         self.assertTrue(d < 1.5)

    #     bug.plot()
    #     bug.plot(path=path)

    # def test_dxform(self):

    #     vars = loadmat("data/map1.mat")
    #     map = vars['map']

    #     dx = DXform(map)
    #     dx.plan([50, 35])
    #     path = dx.query([20, 10])

    #     # valid path
    #     self.assertTrue(path is not None)

    #     # valid Nx2 array
    #     self.assertIsInstance(path, np.ndarray)
    #     self.assertEqual(path.shape[1], 2)

    #     # includes start and goal
    #     self.assertTrue(all(path[0,:] == [20,10]))
    #     self.assertTrue(all(path[-1,:] == [50,35]))

    #     # path doesn't include obstacles
    #     for p in path:
    #         self.assertFalse(dx.is_occupied(p))

    #     # there are no gaps
    #     for k in range(len(path)-1):
    #         d = np.linalg.norm(path[k] - path[k+1])
    #         self.assertTrue(d < 1.5)
            
    #     dx.plot()
    #     dx.plot(path=path)
if __name__ == '__main__':  # pragma nocover

    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
