"""
Created on 28 December 2020
@author: Peter Corke
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import numpy as np
import spatialmath.base as sm
import unittest

# from roboticstoolbox import Bug2, DistanceTransformPlanner, rtb_loadmat
from roboticstoolbox import Bug2
from roboticstoolbox.mobile.Bug2 import edgelist
from roboticstoolbox.mobile.landmarkmap import *
from roboticstoolbox.mobile.drivers import *
from roboticstoolbox.mobile.sensors import *
from roboticstoolbox.mobile.Vehicle import *

# from roboticstoolbox.mobile import Planner

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

        rp = rtb.RandomPath(10)

        self.assertIsInstance(str(rp), str)

        rp.init()

        veh = rtb.Bicycle()

        veh.control = rp

        self.assertIs(veh.control, rp)
        self.assertIs(rp.vehicle, veh)

        u = rp.demand()
        self.assertIsInstance(u, np.ndarray)
        self.assertTrue(u.shape, (2,))


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


if __name__ == "__main__":  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
