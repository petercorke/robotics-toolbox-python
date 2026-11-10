"""
Created on 28 December 2020
@author: Peter Corke
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import numpy as np
import unittest
import pytest

# from roboticstoolbox import Bug2, DistanceTransformPlanner, rtb_loadmat
from roboticstoolbox.mobile.Bug2 import edgelist
from roboticstoolbox.mobile.landmarkmap import *
from roboticstoolbox.mobile.drivers import *
from roboticstoolbox.mobile.sensors import *
from roboticstoolbox.mobile.Vehicle import *
from roboticstoolbox.mobile.Vehicle import DiffSteer, Unicycle
from roboticstoolbox.mobile.ReedsSheppPlanner import ReedsSheppPlanner

# from roboticstoolbox.mobile import Planner

# ======================================================================== #


class TestReedsSheppPlanner(unittest.TestCase):
    def test_turn_around(self):
        start = (0, 0, 0)
        goal = (0, 0, pi)

        reedsshepp = ReedsSheppPlanner(curvature=1.0, stepsize=0.1)
        path, status = reedsshepp.query(start, goal)

        # Turns: LRL and RLR are mirror-symmetric, equal-cost solutions.
        self.assertIn(status[0], (["L", "R", "L"], ["R", "L", "R"]))
        # Total length
        nt.assert_almost_equal(status[1], pi)
        # Segment lengths
        nt.assert_array_almost_equal(status[2], [pi / 3, -pi / 3, pi / 3])


# ======================================================================== #


class TestNavigation(unittest.TestCase):
    def test_edgelist(self):
        im = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        seeds = [(2, 4), (3, 5), (5, 5), (3, 4), (1, 4), (2, 5), (3, 6), (1, 5)]
        for seed in seeds:
            # clockwise
            edge, _ = edgelist(im, seed)
            for e in edge:
                self.assertEqual(im[e[1], e[0]], im[seed[1], seed[0]])

            # counter clockwise
            edge, _ = edgelist(im, seed, -1)
            for e in edge:
                self.assertEqual(im[e[1], e[0]], im[seed[1], seed[0]])

    # def test_map(self):
    #     map = np.zeros((10, 10))
    #     map[2, 3] = 1

    #     # instantiate a noname planner
    #     nav = Planner(occgrid=map, ndims=2)

    #     ## test isoccupied method
    #     self.assertTrue(nav.isoccupied([3, 2]))
    #     self.assertFalse(nav.isoccupied([3, 3]))

    #     # out of bounds
    #     self.assertTrue(nav.isoccupied([20, 20]))

    #     ## test inflation option
    #     nav = Bug2(occgrid=map, inflate=1)
    #     self.assertTrue(nav.isoccupied([3, 2]))
    #     self.assertTrue(nav.isoccupied([3, 3]))
    #     self.assertFalse(nav.isoccupied([3, 4]))


# ======================================================================== #


class RangeBearingSensorTest(unittest.TestCase):
    def setUp(self):
        self.veh = rtb.Bicycle()
        self.map = rtb.LandmarkMap(20)
        self.rs = RangeBearingSensor(self.veh, self.map)

    def test_init(self):

        self.assertIsInstance(self.rs.map, rtb.LandmarkMap)
        self.assertIsInstance(self.rs.robot, rtb.Bicycle)

        self.assertIsInstance(str(self.rs), str)

    def test_animate_deprecated(self):

        with self.assertWarns(DeprecationWarning):
            rs = RangeBearingSensor(self.veh, self.map, animate=True)
        self.assertTrue(rs._animate)

        with self.assertRaises(TypeError):
            RangeBearingSensor(self.veh, self.map, bogus_kwarg=True)

    def test_reading(self):

        z, lm_id = self.rs.reading()
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))

        # test missing samples
        rs = RangeBearingSensor(self.veh, self.map, every=2)

        # first return is (None, None)
        z, lm_id = rs.reading()
        self.assertEqual(z, None)

        z, lm_id = rs.reading()
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))

        z, lm_id = rs.reading()
        self.assertEqual(z, None)

    def test_h(self):
        xv = np.r_[2, 3, 0.5]
        p = np.r_[3, 4]
        z = self.rs.h(xv, 10)
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))
        self.assertAlmostEqual(z[0], np.linalg.norm(self.rs.map[10] - xv[:2]))
        theta = z[1] + xv[2]
        nt.assert_almost_equal(
            self.rs.map[10],
            xv[:2] + z[0] * np.r_[np.cos(theta), np.sin(theta)],
        )

        z = self.rs.h(xv, [3, 4])
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (2,))
        self.assertAlmostEqual(z[0], np.linalg.norm(p - xv[:2]))
        theta = z[1] + 0.5
        nt.assert_almost_equal(
            [3, 4], xv[:2] + z[0] * np.r_[np.cos(theta), np.sin(theta)]
        )

        # all landmarks
        z = self.rs.h(xv)
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (20, 2))
        for k in range(20):
            nt.assert_almost_equal(z[k, :], self.rs.h(xv, k))

        # if vehicle at landmark 10 range=bearing=0
        x = np.r_[self.map[10], 0]
        z = self.rs.h(x, 10)
        self.assertEqual(tuple(z), (0, 0))

        # vectorized forms
        xv = np.array([[2, 3, 0.5], [3, 4, 0], [4, 5, -0.5]])
        z = self.rs.h(xv, 10)
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(z.shape, (3, 2))
        for i in range(3):
            nt.assert_almost_equal(z[i, :], self.rs.h(xv[i, :], 10))

        # xv = np.r_[2, 3, 0.5]
        # p = np.array([[1, 2], [3, 4], [5, 6]]).T
        # z = self.rs.h(xv, p)
        # self.assertIsInstance(z, np.ndarray)
        # self.assertEqual(z.shape, (3,2))
        # for i in range(3):
        #     nt.assert_almost_equal(z[i,:], self.rs.h(xv, p[i,:]))

    def test_H_jacobians(self):
        xv = np.r_[1, 2, pi / 4]
        p = np.r_[5, 7]
        id = 10

        nt.assert_almost_equal(
            self.rs.Hx(xv, id), base.numjac(lambda x: self.rs.h(x, id), xv), decimal=4
        )

        nt.assert_almost_equal(
            self.rs.Hp(xv, p), base.numjac(lambda p: self.rs.h(xv, p), p), decimal=4
        )

        xv = [1, 2, pi / 4]
        p = [5, 7]
        id = 10

        nt.assert_almost_equal(
            self.rs.Hx(xv, id), base.numjac(lambda x: self.rs.h(x, id), xv), decimal=4
        )

        nt.assert_almost_equal(
            self.rs.Hp(xv, p), base.numjac(lambda p: self.rs.h(xv, p), p), decimal=4
        )

    def test_g(self):
        xv = np.r_[1, 2, pi / 4]
        p = np.r_[5, 7]

        z = self.rs.h(xv, p)
        nt.assert_almost_equal(p, self.rs.g(xv, z))

    def test_G_jacobians(self):
        xv = np.r_[1, 2, pi / 4]
        p = np.r_[5, 7]

        z = self.rs.h(xv, p)

        nt.assert_almost_equal(
            self.rs.Gx(xv, z), base.numjac(lambda x: self.rs.g(x, z), xv), decimal=4
        )

        nt.assert_almost_equal(
            self.rs.Gz(xv, z), base.numjac(lambda z: self.rs.g(xv, z), z), decimal=4
        )

    def test_plot(self):

        # map = LandmarkMap(20)
        # map.plot(block=False)
        pass


# ======================================================================== #


class LandMarkTest(unittest.TestCase):
    def test_init(self):

        map = LandmarkMap(20)

        self.assertEqual(len(map), 20)

        lm = map[0]
        self.assertIsInstance(lm, np.ndarray)
        self.assertTrue(lm.shape, (2,))

        self.assertIsInstance(str(lm), str)

    def test_range(self):
        map = LandmarkMap(1000, workspace=[-10, 10, 100, 200])

        self.assertTrue(map._map.shape, (2, 1000))

        for x, y in map:
            self.assertTrue(-10 <= x <= 10)
            self.assertTrue(100 <= y <= 200)

    def test_plot(self):
        plt.clf()
        map = LandmarkMap(20)
        map.plot(block=False)


# ======================================================================== #


class DriversTest(unittest.TestCase):
    def test_init(self):

        rp = rtb.RandomPath(workspace=10)

        self.assertIsInstance(str(rp), str)

        rp.init()

        veh = rtb.Bicycle()

        veh.control = rp

        self.assertIs(veh.control, rp)
        self.assertIs(rp.vehicle, veh)

        u = rp.demand()
        self.assertIsInstance(u, np.ndarray)
        self.assertTrue(u.shape, (2,))

    def test_purepursuit_path(self):
        # a list of lists is converted to a 2xN float array
        pp = rtb.PurePursuit([[0, 10, 10], [0, 0, 10]], lookahead=2)
        self.assertIsInstance(pp._path, np.ndarray)
        self.assertEqual(pp._path.shape, (2, 3))

        # one column per point, so the wrong orientation is an error
        with self.assertRaises(ValueError):
            rtb.PurePursuit([[0, 0], [10, 0], [10, 10]])

        with self.assertRaises(ValueError):
            rtb.PurePursuit([0, 1, 2])

    def test_purepursuit_str(self):
        pp = rtb.PurePursuit([[0, 10], [0, 10]], workspace=[0, 10, -5, 5])
        s = str(pp)
        self.assertIn("X 0.0 : 10.0", s)
        self.assertIn("Y -5.0 : 5.0", s)

        # no workspace
        self.assertIsInstance(str(rtb.PurePursuit([[0, 10], [0, 10]])), str)

    def test_style_merge(self):
        from roboticstoolbox.mobile.drivers import GOAL_STYLE, PATH_STYLE

        # defaults
        rp = rtb.RandomPath(workspace=10)
        self.assertEqual(rp._goal_style, GOAL_STYLE)

        # a user dict is merged over the defaults, and wins
        rp = rtb.RandomPath(workspace=10, goal_style=dict(marker="s", color="b"))
        self.assertEqual(rp._goal_style["marker"], "s")
        self.assertEqual(rp._goal_style["color"], "b")
        self.assertEqual(rp._goal_style["markersize"], GOAL_STYLE["markersize"])

        pp = rtb.PurePursuit([[0, 10], [0, 10]], path_style=dict(linewidth=1))
        self.assertEqual(pp._path_style["linewidth"], 1)
        self.assertEqual(pp._path_style["color"], PATH_STYLE["color"])

        # the module defaults are not modified
        self.assertEqual(GOAL_STYLE["marker"], "D")
        self.assertEqual(PATH_STYLE["linewidth"], 3)

    def test_goalmarkerstyle_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            rp = rtb.RandomPath(workspace=10, goalmarkerstyle=dict(marker="s"))
        # the old name still works
        self.assertEqual(rp._goal_style["marker"], "s")

        # the new name wins if both are given
        with self.assertWarns(DeprecationWarning):
            rp = rtb.RandomPath(
                workspace=10,
                goalmarkerstyle=dict(marker="s"),
                goal_style=dict(marker="o"),
            )
        self.assertEqual(rp._goal_style["marker"], "o")

    def test_purepursuit_run(self):
        # without animation, there is no goal marker
        path = np.array([[0.0, 10, 10], [0, 0, 10]])
        veh = rtb.Bicycle(control=rtb.PurePursuit(path, lookahead=2, speed=2))
        veh.run(5, animate=False)
        self.assertIsNone(veh.control._goal_marker)

        u = veh.control.demand()
        self.assertEqual(u.shape, (2,))

    def test_goal_marker(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # a dense path, as produced by a trajectory generator
        xs = np.arange(0, 50, 0.5)
        path = np.vstack((xs, np.zeros_like(xs)))
        pp = rtb.PurePursuit(path, lookahead=3, speed=2)
        veh = rtb.Bicycle(control=pp)
        veh.init(animate=False)

        fig, ax = plt.subplots()
        nlines = len(ax.lines)
        pp.init(ax=ax)
        # the path and a single goal marker
        self.assertEqual(len(ax.lines), nlines + 2)

        # the goal marker moves, rather than leaving a trail
        goals = []
        for _ in range(50):
            pp.demand()
            goals.append(pp._goal_marker.get_xydata().copy())
            veh.step()
        self.assertEqual(len(ax.lines), nlines + 2)
        self.assertFalse(np.allclose(goals[0], goals[-1]))
        plt.close(fig)

    def test_randompath_goal_marker(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rp = rtb.RandomPath(workspace=10)
        veh = rtb.Bicycle(control=rp)
        veh.init(animate=False)

        fig, ax = plt.subplots()
        rp.init(ax=ax)
        rp.demand()
        self.assertEqual(len(ax.lines), 1)
        np.testing.assert_allclose(rp._goal_marker.get_xydata()[0], rp._goal)
        plt.close(fig)


class TestVehicleBase(unittest.TestCase):
    def test_constructor_control_constant(self):
        veh = rtb.Bicycle(control=(1.0, 0.1))
        self.assertEqual(veh.control, (1.0, 0.1))
        u = veh.eval_control(veh.control, veh.x)
        nt.assert_allclose(u, [1.0, 0.1])

    def test_constructor_control_function(self):
        veh = rtb.Bicycle(control=lambda v, t, x: (2.0, 0.0))
        nt.assert_allclose(veh.eval_control(veh.control, veh.x), [2.0, 0.0])

    def test_constructor_control_driver(self):
        # a driver is connected without the deprecation warning
        import warnings

        rp = rtb.RandomPath(workspace=10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            veh = rtb.Bicycle(control=rp)
        self.assertIs(veh.control, rp)
        self.assertIs(rp.vehicle, veh)

    def test_init_control_driver(self):
        # a driver given to init/run is connected to the vehicle
        veh = rtb.Bicycle()
        rp = rtb.RandomPath(workspace=10)
        veh.init(control=rp, animate=False)
        self.assertIs(rp.vehicle, veh)
        self.assertEqual(veh.run(1, control=rp, animate=False).shape[1], 3)

    def test_bad_control(self):
        veh = rtb.Bicycle()
        with self.assertRaises(ValueError):
            veh.eval_control("bad", veh.x)

    def test_polygon(self):
        veh = rtb.Bicycle()
        with self.assertRaises(ValueError):
            veh.polygon([0, 0, 0])

    def test_run_animation_file_required(self):
        veh = rtb.Bicycle(control=(1.0, 0.0))
        for fmt in ("gif", "mp4"):
            with self.assertRaises(ValueError):
                veh.run_animation(T=1, format=fmt)
        with self.assertRaises(ValueError):
            veh.run_animation(T=1, format="bad")

    def test_plot_xyt(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        veh = rtb.Bicycle(control=(1.0, 0.1), dt=0.1)
        veh.run(2, animate=False)
        n = veh.x_hist.shape[0]

        fig = plt.figure()
        veh.plot_xyt()
        lines = plt.gca().lines
        self.assertEqual(len(lines), 3)
        t = lines[0].get_xdata()
        self.assertEqual(len(t), n)
        nt.assert_allclose(t, np.arange(n) * 0.1)
        # the y-data are x, y, theta
        nt.assert_allclose(lines[2].get_ydata(), veh.x_hist[:, 2])
        plt.close(fig)

    def test_plot_xy(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        veh = rtb.Bicycle(control=(1.0, 0.1))
        veh.run(1, animate=False)
        fig = plt.figure()
        veh.plot_xy()
        veh.plot_xy("r--", label="path")
        self.assertEqual(len(plt.gca().lines), 2)
        plt.close(fig)


class TestBicycle(unittest.TestCase):
    # def test_deriv(self):
    #     xv = np.r_[1, 2, pi/4]

    #     veh = Bicycle()

    #     u = [1, 0.2]
    #     nt.assert_almost_equal(
    #         veh.deriv(xv, ),
    #         base.numjac(lambda p: veh(xv, p), p),
    #         decimal=4)

    def test_jacobians(self):
        xv = np.r_[1, 2, pi / 4]
        odo = np.r_[0.1, 0.2]
        veh = Bicycle()

        nt.assert_almost_equal(
            veh.Fx(xv, odo), base.numjac(lambda x: veh.f(x, odo), xv), decimal=4
        )

        nt.assert_almost_equal(
            veh.Fv(xv, odo), base.numjac(lambda d: veh.f(xv, d), odo), decimal=4
        )


@pytest.mark.parametrize("width", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("turn_rate", [-0.4, 0.4])
def test_unicycle_turn_rate_units(width: float, turn_rate: float) -> None:
    vehicle = Unicycle(W=width)
    state = np.array([1.0, 2.0, np.pi / 4])
    control = [0.7, turn_rate]
    expected = [0.7 / np.sqrt(2), 0.7 / np.sqrt(2), turn_rate]
    nt.assert_allclose(vehicle.deriv(state, control, limits=False), expected)
    nt.assert_allclose(vehicle.deriv(state, control), expected)


@pytest.mark.parametrize("width", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("turn_rate", [-0.4, 0.4])
def test_unicycle_turn_rate_limit_step(width: float, turn_rate: float) -> None:
    vehicle = Unicycle(W=width, steer_max=0.3, dt=0.1)
    expected_turn_rate = np.copysign(0.3, turn_rate)
    for step in range(1, 4):
        odometry = vehicle.step([0.0, turn_rate], animate=False)
        nt.assert_allclose(odometry, [0.0, expected_turn_rate * 0.1])
        nt.assert_allclose(vehicle.x, [0.0, 0.0, step * expected_turn_rate * 0.1])


@pytest.mark.parametrize("width", [0.5, 1.0, 2.0])
def test_diffsteer_wheel_speed_units(width: float) -> None:
    vehicle = DiffSteer(W=width)
    state = np.array([1.0, 2.0, 0.0])
    nt.assert_allclose(vehicle.deriv(state, [1.0, 2.0]), [1.5, 0.0, 1.0 / width])


class TestUnicycle(unittest.TestCase):
    def test_str(self):
        """
        check the string representation of the unicycle
        """
        uni = Unicycle()
        self.assertEqual(
            str(uni),
            """Unicycle: x = [ 0, 0, 0 ]
  W=1, steer_max=inf, vel_max=inf, accel_max=inf""",
        )

        uni = Unicycle(steer_max=0.7)
        self.assertEqual(
            str(uni),
            """Unicycle: x = [ 0, 0, 0 ]
  W=1, steer_max=0.7, vel_max=inf, accel_max=inf""",
        )

    def test_deriv(self):
        """
        test the derivative function
        """
        uni = Unicycle()

        state = np.r_[0, 0, 0]
        input = [1, 0]  # no rotation
        nt.assert_almost_equal(uni.deriv(state, input), np.r_[1.0, 0, 0])

        input = [0, 1]  # only rotate
        nt.assert_almost_equal(uni.deriv(state, input), np.r_[0, 0, 1])

        input = [1, 1]  # turn and rotate
        nt.assert_almost_equal(uni.deriv(state, input), np.r_[1, 0, 1])


# function setupOnce(testCase)
#     testCase.TestData.Duration = 50;
# end

# function Vehicle_test(tc)
#     %%
#     randinit
#     V = diag([0.005, 0.5*pi/180].^2);

#     v = Bicycle('covar', V);
#     v.add_driver( RandomPath(10) );

#     v.run(tc.TestData.Duration);
#     v.plot_xy();
#     s = v.char();

#     J = v.Fx(v.x, [.1 .2]);
#     J = v.Fv(v.x, [.1 .2]);
# end

# function DeadReckoning_test(tc)
#     %%
#     randinit
#     V = diag([0.005, 0.5*pi/180].^2);
#     P0 = diag([0.005, 0.005, 0.001].^2);

#     v = Bicycle('covar', V);
#     v.add_driver( RandomPath(10) );
#     s = char(v);

#     ekf = EKF(v, V, P0);
#     ekf.run(tc.TestData.Duration);

#     clf
#     ekf.plot_xy
#     hold on
#     v.plot_xy('r')
#     grid on
#     xyzlabel

#     ekf.plot_ellipse('g')
#     ekf.plot_P()
# end

# function MapLocalization_test(tc)
#     %%
#     randinit
#     W = diag([0.1, 1*pi/180].^2);
#     P0 = diag([0.005, 0.005, 0.001].^2);
#     V = diag([0.005, 0.5*pi/180].^2);

#     map = LandmarkMap(20);
#     map = LandmarkMap(20, 'verbose');
#     map = LandmarkMap(20, 10, 'verbose');
#     map = LandmarkMap(20, 10);
#     s = char(map);

#     veh = Bicycle('covar', V);
#     veh.add_driver( RandomPath(10) );
#     sensor = RangeBearingSensor(veh, map, 'covar', W);
#     sensor.interval = 5;
#     ekf = EKF(veh, W, P0, sensor, W, map);

#     ekf.run(tc.TestData.Duration);

#     clf
#     map.plot()
#     veh.plot_xy('b');
#     ekf.plot_xy('r');
#     ekf.plot_ellipse('k')
#     grid on
#     xyzlabel

#     clf
#     ekf.plot_P()
# end

# function Mapping_test(tc)
#     %%
#     randinit
#     W = diag([0.1, 1*pi/180].^2);
#     V = diag([0.005, 0.5*pi/180].^2);

#     map = LandmarkMap(20, 10);

#     veh = Bicycle('covar', V);
#     veh.add_driver( RandomPath(10) );

#     sensor = RangeBearingSensor(veh, map, 'covar', W);
#     sensor.interval = 5;

#     ekf = EKF(veh, [], [], sensor, W, []);
#     ekf.run(tc.TestData.Duration);


#     clf
#     map.plot()
#     veh.plot_xy('b');
#     ekf.plot_map('g');
#     grid on
#     xyzlabel

#     %%
#     verifyEqual(tc, numcols(ekf.landmarks), 20);

# end

# function SLAM_test(tc)
#     %%
#     randinit
#     W = diag([0.1, 1*pi/180].^2);
#     P0 = diag([0.005, 0.005, 0.001].^2);
#     V = diag([0.005, 0.5*pi/180].^2);

#     map = LandmarkMap(20, 10);

#     veh = Bicycle(V);
#     veh.add_driver( RandomPath(10) );

#     sensor = RangeBearingSensor(veh, map, 'covar', W);
#     sensor.interval = 1;

#     ekf = EKF(veh, V, P0, sensor, W, []);
#     ekf
#     ekf.verbose = false;
#     ekf.run(tc.TestData.Duration);


#     clf
#     map.plot()
#     veh.plot_xy('b');
#     ekf.plot_xy('r');
#     ekf.plot_ellipse('k')
#     grid on
#     xyzlabel

#     clf
#     ekf.plot_P()

#     clf
#     map.plot();
#     ekf.plot_map('g');

#     %%
#     verifyEqual(tc, numcols(ekf.landmarks), 20);

# end

# function ParticleFilter_test(tc)
#     %%
#     randinit
#     map = LandmarkMap(20);

#     W = diag([0.1, 1*pi/180].^2);
#     v = Bicycle('covar', W);
#     v.add_driver( RandomPath(10) );
#     V = diag([0.005, 0.5*pi/180].^2);
#     sensor = RangeBearingSensor(v, map, 'covar', V);

#     Q = diag([0.1, 0.1, 1*pi/180]).^2;
#     L = diag([0.1 0.1]);
#     pf = ParticleFilter(v, sensor, Q, L, 1000);
#     pf
#     pf.run(tc.TestData.Duration);

#     plot(pf.std)
#     xlabel('time step')
#     ylabel('standard deviation')
#     legend('x', 'y', '\theta')
#     grid

#     clf
#     pf.plot_pdf();
#     clf
#     pf.plot_xy();
# end

# function posegraph_test(tc)
#     pg = PoseGraph('pg1.g2o')
#     self.assertClass(pg, 'PoseGraph');
#     self.assertEqual(pg.graph.n, 4);

#     clf
#     pg.plot()
#     pg.optimize('animate')
#     close all

#     pg = PoseGraph('killian-small.toro')
#     self.assertClass(pg, 'PoseGraph');
#     self.assertEqual(pg.graph.n, 1941);

#     pg = PoseGraph('killian.g2o', 'laser')
#     self.assertClass(pg, 'PoseGraph');
#     self.assertEqual(pg.graph.n, 3873);

#     [r,theta] = pg.scan(1);
#     self.assertClass(r, 'double');
#     self.assertLength(r, 180);
#     self.assertClass(theta, 'double');
#     self.assertLength(theta, 180);

#     [x,y] = pg.scan(1);
#     self.assertClass(x, 'double');
#     self.assertLength(x, 180);
#     self.assertClass(y, 'double');
#     self.assertLength(y, 180);

#     pose = pg.pose(1);
#     self.assertClass(pose, 'double');
#     self.assertSize(pose, [3 1]);

#     t = pg.time(1);
#     self.assertClass(t, 'double');
#     self.assertSize(t, [1 1]);

#     w = pg.scanmap('ngrid', 3000);
#     self.assertClass(w, 'int32');
#     self.assertSize(w, [3000 3000]);

#     tc.assumeTrue(exist('idisp', 'file'));  %REMINDER
#     clf
#     pg.plot_occgrid(w);
#     close all

# end

# function makemap_test(tc)
#         tc.assumeTrue(false);  %REMINDER

# end

# function chi2inv_test(tc)
#     self.assertEqual( chi2inv_rtb(0,2), 0);
#     self.assertEqual( chi2inv_rtb(1,2), Inf);
#     self.assertEqual( chi2inv_rtb(3,2), NaN);

#     self.assertError( @() chi2inv_rtb(1,1), 'RTB:chi2inv_rtb:badarg');

# end
#


class TestRandomPathWorkspace(unittest.TestCase):
    def test_workspace_is_required(self):
        # the driver chooses waypoints inside the workspace, so it needs one,
        # its own or that of the vehicle it is attached to.  It is checked when
        # the simulation is initialized; this used to fail with an obscure
        # TypeError on None
        driver = rtb.RandomPath()
        veh = rtb.Bicycle()
        veh.control = driver
        with self.assertRaises(ValueError):
            veh.init(animate=False)

    def test_workspace_given(self):
        driver = rtb.RandomPath(workspace=10)
        self.assertIsNotNone(driver)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
