#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Peter Corke
"""
from roboticstoolbox.tools.trajectory import *
import numpy as np
import numpy.testing as nt
import unittest
from spatialmath import SE3
from math import pi

_eps = np.finfo(np.float64).eps

class TestTrajectory(unittest.TestCase):

    def test_tpoly(self):

        s1 = 1
        s2 = 2
        ## no boundary conditions
                
        tg = tpoly(s1, s2, 11)
        s = tg.y
        sd = tg.yd
        sdd = tg.ydd

        self.assertTrue(np.all(np.diff(s) > 0))  # output is monotonic
        self.assertAlmostEqual(s[0], s1)
        self.assertAlmostEqual(s[-1], s2)
        self.assertAlmostEqual(s[5], 1.5)

        self.assertTrue(np.all(sd >= -10 * _eps))  # velocity is >= 0
        self.assertAlmostEqual(sd[0], 0)
        self.assertAlmostEqual(sd[-1], 0)
        
        self.assertAlmostEqual(sdd[0], 0)
        self.assertAlmostEqual(sdd[5], 0)
        self.assertAlmostEqual(sdd[-1], 0)
        self.assertAlmostEqual(sum(sdd), 0)
        
        ## time vector version
        t = np.linspace(0, 1, 11)
        
        tg = tpoly(s1, s2, t)
        s = tg.y
        sd = tg.yd
        sdd = tg.ydd

        self.assertTrue(np.all(np.diff(s) > 0))  # output is monotonic
        self.assertAlmostEqual(s[0], s1)
        self.assertAlmostEqual(s[-1], s2)
        self.assertAlmostEqual(s[5], 1.5)

        self.assertTrue(np.all(sd >= -100 * _eps))  # velocity is >= 0, some numeric issues hence the possible small negative allowance
        self.assertAlmostEqual(sd[0], 0)
        self.assertAlmostEqual(sd[-1], 0)
        
        self.assertAlmostEqual(sdd[0], 0)
        self.assertAlmostEqual(sdd[5], 0)
        self.assertAlmostEqual(sdd[-1], 0)
        self.assertAlmostEqual(sum(sdd), 0)

        ## boundary conditions
        tg = tpoly(s1, s2, 11, -1, 1)
        s = tg.y
        sd = tg.yd
        sdd = tg.ydd

        self.assertAlmostEqual(s[0], s1)
        self.assertAlmostEqual(s[-1], s2)
        
        self.assertAlmostEqual(sd[0], -1)
        self.assertAlmostEqual(sd[-1], 1)
        
        self.assertAlmostEqual(sdd[0], 0)
        self.assertAlmostEqual(sdd[-1], 0)
                   
    def test_lspb(self):

        s1 = 1
        s2 = 2
        
        ## no boundary conditions
        
        tg = lspb(s1, s2, 11)
        s = tg.y
        sd = tg.yd
        sdd = tg.ydd

        self.assertTrue(np.all(np.diff(s) > 0))  # output is monotonic
        self.assertAlmostEqual(s[0], s1)
        self.assertAlmostEqual(s[-1], s2)
        self.assertAlmostEqual(s[5], 1.5)

        self.assertTrue(np.all(sd >= -10 * _eps))  # velocity is >= 0
        self.assertAlmostEqual(sd[0], 0)
        self.assertAlmostEqual(sd[-1], 0)
  
        self.assertAlmostEqual(np.sum(sdd), 0)

        ## time vector version
        t = np.linspace(0, 1, 11)
        
        tg = lspb(s1, s2, t)
        s = tg.y
        sd = tg.yd
        sdd = tg.ydd
        
        self.assertTrue(np.all(np.diff(s) > 0))  # output is monotonic
        self.assertAlmostEqual(s[0], s1)
        self.assertAlmostEqual(s[-1], s2)
        self.assertAlmostEqual(s[5], 1.5)

        self.assertTrue(np.all(sd >= -10 * _eps))  # velocity is >= 0
        self.assertAlmostEqual(sd[0], 0)
        self.assertAlmostEqual(sd[-1], 0)
  
        self.assertAlmostEqual(np.sum(sdd), 0)
            
        ## specify velocity
        tg = lspb(s1, s2, 11,0.2)
        s = tg.y
        sd = tg.yd
        sdd = tg.ydd

        self.assertAlmostEqual(s[0], s1)
        self.assertAlmostEqual(s[-1], s2)
        
        self.assertAlmostEqual(sd[5], 0.2)

    def test_tg1plot(self):
        s1 = 1
        s2 = 2

        tg = tpoly(s1, s2, 11)
        t1plot(tg, block=False)

        tg = lspb(s1, s2, 11)
        t1plot(tg, block=False)

    def test_qplot(self):

        # 6 joints is special
        q1 = np.r_[1, 2, 3, 4, 5, 6]
        q2 = -q1
        q = jtraj(q1, q2, 50)
        qplot(q.q, block=False)

        # 4 joints
        q1 = np.r_[1, 2, 3, 4]
        q2 = -q1
        q = jtraj(q1, q2, 50)
        qplot(q.q, block=False)
 
            
    def test_ctraj(self):
        # unit testing ctraj with T0 and T1 and N
        T0 = SE3(1,2,3)
        T1 = SE3(-1,-2,-3)
        
        T = ctraj(T0, T1, 3)
        self.assertEqual(len(T), 3)
        nt.assert_array_almost_equal(T[0].A, T0.A)
        nt.assert_array_almost_equal(T[2].A, T1.A)
        nt.assert_array_almost_equal(T[1].A, SE3().A)

        # unit testing ctraj with T0 and T1 and S[i]
        T = ctraj(T0, T1, [1, 0, 0.5])
        self.assertEqual(len(T), 3)

        nt.assert_array_almost_equal(T[0].A, T1.A)
        nt.assert_array_almost_equal(T[1].A, T0.A)
        nt.assert_array_almost_equal(T[2].A, SE3().A)

        T0 = SE3.Rx(-pi/2)
        T1 = SE3.Rx(pi/2)
        
        T = ctraj(T0, T1, 3)
        self.assertEqual(len(T), 3)
        nt.assert_array_almost_equal(T[0].A, T0.A)
        nt.assert_array_almost_equal(T[2].A, T1.A)
        nt.assert_array_almost_equal(T[1].A, SE3().A)

    def test_jtraj(self):
        # unit testing jtraj with 
        q1 = np.r_[1, 2, 3, 4, 5, 6]
        q2 = -q1
        
        tg = jtraj(q1, q2, 11)
        q = tg.q
        qd = tg.qd
        qdd = tg.qdd

        self.assertAlmostEqual(q.shape, (11,6))
        self.assertTrue(np.allclose(q[0,:], q1))
        self.assertTrue(np.allclose(q[-1,:], q2))
        self.assertTrue(np.allclose(q[5,:], np.zeros(6,)))

        self.assertAlmostEqual(qd.shape, (11,6))
        self.assertTrue(np.allclose(qd[0,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qd[-1,:], np.zeros(6,)))

        self.assertAlmostEqual(qdd.shape, (11,6))
        self.assertTrue(np.allclose(qdd[0,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qdd[-1,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qdd[5,:], np.zeros(6,)))
        

        # with a time vector
        t = np.linspace(0, 2, 11)
        
        tg = jtraj(q1, q2, t)
        q = tg.q
        qd = tg.qd
        qdd = tg.qdd

        self.assertAlmostEqual(q.shape, (11,6))
        self.assertTrue(np.allclose(q[0,:], q1))
        self.assertTrue(np.allclose(q[-1,:], q2))
        self.assertTrue(np.allclose(q[5,:], np.zeros(6,)))

        self.assertAlmostEqual(qd.shape, (11,6))
        self.assertTrue(np.allclose(qd[0,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qd[-1,:], np.zeros(6,)))

        self.assertAlmostEqual(qdd.shape, (11,6))
        self.assertTrue(np.allclose(qdd[0,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qdd[-1,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qdd[5,:], np.zeros(6,)))
        
        # test with boundary conditions
        qone = np.ones((6,))
        tg = jtraj(q1, q2, 11, -qone, qone)
        q = tg.q
        qd = tg.qd
        qdd = tg.qdd

        self.assertAlmostEqual(q.shape, (11,6))
        self.assertTrue(np.allclose(q[0,:], q1))
        self.assertTrue(np.allclose(q[-1,:], q2))

        self.assertAlmostEqual(qd.shape, (11,6))
        self.assertTrue(np.allclose(qd[0,:], -qone))
        self.assertTrue(np.allclose(qd[-1,:], qone))

        self.assertAlmostEqual(qdd.shape, (11,6))
        self.assertTrue(np.allclose(qdd[0,:], np.zeros(6,)))
        self.assertTrue(np.allclose(qdd[-1,:], np.zeros(6,)))
        
    def test_mstraj(self):

        via = np.array([
            [4, 1],
            [4, 4],
            [5, 2],
            [2, 5]
            ])

        #Test with QDMAX 
        out = mstraj(via, dt=1, tacc=1, qdmax=[2, 1], q0=[4, 1])

        # expected_out = mstraj(via, [ 2 1 ],[],[4 1],1,1,1);
        expected_out = np.array([
                        [4.0000,    1.0000],
                        [4.0000 ,   1.7500],
                        [4.0000,    2.5000],
                        [4.0000,    3.2500],
                        [4.3333,    3.3333],
                        [4.6667,    2.6667],
                        [4.2500,    2.7500],
                        [3.5000,    3.5000],
                        [2.7500,    4.2500],
                        [2.0000,    5.0000]
                        ])
        nt.assert_array_almost_equal(out.q, expected_out, decimal=4)

         #Test with QO 
         # expected_out = mstraj(via, [], [2 1 3 4],[4 1],1,1,1);
        out = mstraj(via, dt=1, tacc=1, tsegment=[2, 1, 3, 4], q0=[4, 1])
        expected_out = np.array([
                        [4.0000,    1.0000],
                        [4.0000,    4.0000],
                        [4.3333,    3.3333],
                        [4.6667,    2.6667],
                        [4.2500,    2.7500],
                        [3.5000,    3.5000],
                        [2.7500,    4.2500],
                        [2.0000,    5.0000]
                        ])
        nt.assert_array_almost_equal(out.q, expected_out, decimal=4)
         
        out = mstraj(via, dt=1, tacc=1, tsegment=[1, 2, 3, 4], q0=via[0,:])
        self.assertEqual(out.t.shape[0], out.q.shape[0])
         
        self.assertIsInstance(out.info, list)
        self.assertEqual(len(out.info), via.shape[0]+1)


if __name__ == '__main__':

    unittest.main()


    # function mtraj_tpoly_test(tc)
    #     q1 = [1 2 3 4 5 6]
    #     q2 = -q1
        
    #     q = mtraj(@tpoly, q1,q2,11)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     [q,qd] = mtraj(@tpoly, q1,q2,11)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     [q,qd,qdd] = mtraj(@tpoly, q1,q2,11)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qdd,1), 11)
    #     self.assertAlmostEqual(size(qdd,2), 6)
    #     self.assertAlmostEqual(qdd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qdd(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(qdd(end,:), zeros(1,6))
        
    #     ## with a time vector
    #     t = linspace(0, 1, 11)
        
    #     q = mtraj(@tpoly, q1,q2,t)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     [q,qd] = mtraj(@tpoly, q1,q2,t)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     [q,qd,qdd] = mtraj(@tpoly, q1,q2,t)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qdd,1), 11)
    #     self.assertAlmostEqual(size(qdd,2), 6)
    #     self.assertAlmostEqual(qdd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qdd(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(qdd(end,:), zeros(1,6))
    # end

    # function mtraj_lspb_test(tc)
    #     q1 = [1 2 3 4 5 6]
    #     q2 = -q1
        
    #     q = mtraj(@lspb, q1,q2,11)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     [q,qd] = mtraj(@lspb, q1,q2,11)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     [q,qd,qdd] = mtraj(@lspb, q1,q2,11)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qdd,1), 11)
    #     self.assertAlmostEqual(size(qdd,2), 6)
        
    #     self.assertAlmostEqual(sum(qdd), zeros(1,6))
        
    #     ## with a time vector
    #     t = linspace(0, 1, 11)
        
    #     q = mtraj(@lspb, q1,q2,t)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     [q,qd] = mtraj(@lspb, q1,q2,t)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     [q,qd,qdd] = mtraj(@lspb, q1,q2,t)
    #     self.assertAlmostEqual(size(q,1), 11)
    #     self.assertAlmostEqual(size(q,2), 6)
    #     self.assertAlmostEqual(q(1,:), q1)
    #     self.assertAlmostEqual(q(end,:), q2)
    #     self.assertAlmostEqual(q(6,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qd,1), 11)
    #     self.assertAlmostEqual(size(qd,2), 6)
    #     self.assertAlmostEqual(qd(1,:), zeros(1,6))
    #     self.assertAlmostEqual(qd(end,:), zeros(1,6))
        
    #     self.assertAlmostEqual(size(qdd,1), 11)
    #     self.assertAlmostEqual(size(qdd,2), 6)
        
    #     self.assertAlmostEqual(sum(qdd), zeros(1,6))

