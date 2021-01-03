
"""
Created on 28 December 2020
@author: Peter Corke
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import numpy as np
import spatialmath.base as sm
import unittest

from roboticstoolbox import Bug2, DXform, loadmat
from roboticstoolbox.mobile.bug2 import edgelist

class TestNavigation(unittest.TestCase):

    def test_edgelist(self):
        im = np.array([
            [0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    1,    1,    1,    0,    0],
            [0,    0,    0,    1,    1,    1,    0,    0],
            [0,    0,    1,    1,    1,    1,    0,    0],
            [0,    0,    0,    1,    1,    1,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,    0],
            [0,    0,    0,    0,    0,    0,    0,    0],
        ])
        
        seeds = [(2, 4), (3, 5), (5, 5), (3, 4), (1, 4), (2, 5), (3, 6), (1, 5)]
        for seed in seeds:
                # clockwise
                edge, _ = edgelist(im, seed)
                for e in edge:
                    self.assertEqual( im[e[1],e[0]], im[seed[1], seed[0]] )
               
                # counter clockwise
                edge, _ = edgelist(im, seed, -1);
                for e in edge:
                    self.assertEqual( im[e[1],e[0]], im[seed[1], seed[0]] )


    def test_map(self):
        map = np.zeros((10,10))
        map[2,3] = 1
        
        nav = Bug2(map) # we can't instantiate Navigation because it's abstract
        
        ## test isoccupied method
        self.assertTrue( nav.is_occupied([3,2]) )
        self.assertFalse( nav.is_occupied([3,3]) )
    
        
        # out of bound
        self.assertTrue( nav.is_occupied([20, 20]) )
        
        
        ## test inflation option
        nav = Bug2(map, inflate=1);
        self.assertTrue( nav.is_occupied([3,2]) )
        self.assertTrue( nav.is_occupied([3,3]) )
        self.assertFalse( nav.is_occupied([3,4]) )


    def test_rand(self):
    
        og = np.r_[0]
        nav = Bug2(og)  # we can't instantiate Navigation because it's abstract
        
        # test random number generator
        r = nav.rand()
        self.assertIsInstance(r, float)
        self.assertTrue(0 <= r <= 1)

        r = nav.rand(low=10, high=20);
        self.assertIsInstance(r, float)
        self.assertTrue(10 <= r <= 20)

        r = nav.rand(size=(3,4))
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, (3,4))


    def test_randn(self):
    
        og = np.r_[0]
        nav = Bug2(og)  # we can't instantiate Navigation because it's abstract
        
        # test random number generator
        r = nav.randn()
        self.assertIsInstance(r, float)

        r = nav.randn(size=(3,4))
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, (3,4))

        r = nav.randn(size=(100,))
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, (100,))
        self.assertTrue(-0.4 <= np.mean(r) <= 0.4)

        r = nav.randn(loc=100, size=(100,))
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, (100,))
        self.assertTrue(99 <= np.mean(r) <= 101)

    def test_randi(self):
    
        og = np.r_[0]
        nav = Bug2(og)  # we can't instantiate Navigation because it's abstract
        
        # test random number generator
        r = nav.randi(10)
        self.assertIsInstance(r, np.int64)
        self.assertTrue(0 <= r < 10)

        r = nav.randi(low=10, high=20);
        self.assertIsInstance(r, np.int64)
        self.assertTrue(10 <= r < 20)

        r = nav.randi(10, size=(3,4))
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(r.shape, (3,4))

    def test_bug2(self):

        vars = loadmat("data/map1.mat")
        map = vars['map']

        bug = Bug2(map)
        # bug.plan()
        path = bug.query([20, 10], [50, 35])

        # valid path
        self.assertTrue(path is not None)

        # valid Nx2 array
        self.assertIsInstance(path, np.ndarray)
        self.assertEqual(path.shape[1], 2)

        # includes start and goal
        self.assertTrue(all(path[0,:] == [20,10]))
        self.assertTrue(all(path[-1,:] == [50,35]))

        # path doesn't include obstacles
        for p in path:
            self.assertFalse(bug.is_occupied(p))

        # there are no gaps
        for k in range(len(path)-1):
            d = np.linalg.norm(path[k] - path[k+1])
            self.assertTrue(d < 1.5)

        bug.plot()
        bug.plot(path=path)

    def test_dxform(self):

        vars = loadmat("data/map1.mat")
        map = vars['map']

        dx = DXform(map)
        dx.plan([50, 35])
        path = dx.query([20, 10])

        # valid path
        self.assertTrue(path is not None)

        # valid Nx2 array
        self.assertIsInstance(path, np.ndarray)
        self.assertEqual(path.shape[1], 2)

        # includes start and goal
        self.assertTrue(all(path[0,:] == [20,10]))
        self.assertTrue(all(path[-1,:] == [50,35]))

        # path doesn't include obstacles
        for p in path:
            self.assertFalse(dx.is_occupied(p))

        # there are no gaps
        for k in range(len(path)-1):
            d = np.linalg.norm(path[k] - path[k+1])
            self.assertTrue(d < 1.5)
            
        dx.plot()
        dx.plot(path=path)

class TestVehicle(unittest.TestCase):
	pass
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



if __name__ == '__main__':  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])