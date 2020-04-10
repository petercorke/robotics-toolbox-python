#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep
"""

# Some unit tests

import numpy.testing as nt
import unittest

from transforms import *
        
class Test2D(unittest.TestCase):
    def test_rot2(self):
        R = np.array([[1, 0], [0, 1]])
        nt.assert_array_almost_equal(rot2(0),  R)
        nt.assert_array_almost_equal(rot2(0, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(0, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(0, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(0)), 1)
        
        R = np.array([[0, -1], [1, 0]])
        nt.assert_array_almost_equal(rot2(math.pi/2),  R)
        nt.assert_array_almost_equal(rot2(math.pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(90, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(90, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(math.pi/2)), 1)
        
        R = np.array([[-1, 0], [0, -1]])
        nt.assert_array_almost_equal(rot2(math.pi), R)
        nt.assert_array_almost_equal(rot2(math.pi, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(180, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(math.pi)), 1)
        
    def test_trot2(self):
        nt.assert_array_almost_equal(trot2(math.pi/2, t=[3,4]),  np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
        nt.assert_array_almost_equal(trot2(math.pi/2, t=(3,4)),  np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
        nt.assert_array_almost_equal(trot2(math.pi/2, t=np.array([3,4])),  np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
    
    def test_Rt(self):
        nt.assert_array_almost_equal(rot2(0.3), t2r(trot2(0.3)))
        nt.assert_array_almost_equal(trot2(0.3), r2t(rot2(0.3)))
        
        R = rot2(0.2)
        t = [1, 2]
        T = rt2tr(R,t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl2(T), np.array(t))
        # TODO
        
    def test_transl2(self):
        nt.assert_array_almost_equal(transl2(1, 2), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]]) )
        # TODO
        
    def test_checks(self):
        # 2D case, with rotation matrix
        R = np.eye(2)
        nt.assert_equal( isR(R),          True )
        nt.assert_equal( isrot2(R),       True )
        nt.assert_equal( isrot(R),        False )
        nt.assert_equal( ishom(R),        False)
        nt.assert_equal( ishom2(R),       False )
        nt.assert_equal( isrot2(R, True), True )
        nt.assert_equal( isrot(R, True),  False )
        nt.assert_equal( ishom(R, True),  False )
        nt.assert_equal( ishom2(R, True), False )
        
        # 2D case, invalid rotation matrix
        R = np.array([[1,1], [0,1]])
        nt.assert_equal( isR(R),          False )
        nt.assert_equal( isrot2(R),       True )
        nt.assert_equal( isrot(R),        False )
        nt.assert_equal( ishom(R),        False)
        nt.assert_equal( ishom2(R),       False )
        nt.assert_equal( isrot2(R, True), False )
        nt.assert_equal( isrot(R, True),  False )
        nt.assert_equal( ishom(R, True),  False )
        nt.assert_equal( ishom2(R, True), False )
        
        # 2D case, with homogeneous transformation matrix
        T = np.array([[1,0, 3],[0, 1, 4], [0, 0, 1]])
        nt.assert_equal( isR(T),          False )
        nt.assert_equal( isrot2(T),       False )
        nt.assert_equal( isrot(T),        True )
        nt.assert_equal( ishom(T),        False)
        nt.assert_equal( ishom2(T),       True )
        nt.assert_equal( isrot2(T, True), False )
        nt.assert_equal( isrot(T, True),  False )
        nt.assert_equal( ishom(T, True),  False )
        nt.assert_equal( ishom2(T, True), True )
        
        # 2D case, ivalid rotation matrix
        T = np.array([[1,1, 3],[0, 1, 4], [0, 0, 1]])
        nt.assert_equal( isR(T),          False )
        nt.assert_equal( isrot2(T),       False )
        nt.assert_equal( isrot(T),        True )
        nt.assert_equal( ishom(T),        False)
        nt.assert_equal( ishom2(T),       True )
        nt.assert_equal( isrot2(T, True), False )
        nt.assert_equal( isrot(T, True),  False )
        nt.assert_equal( ishom(T, True),  False )
        nt.assert_equal( ishom2(T, True), False )
        
        
class Test3D(unittest.TestCase):
    
    def test_rotx(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotx(0),  R)
        nt.assert_array_almost_equal(rotx(0, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(0, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(0, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotx(0)), 1)
        
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        nt.assert_array_almost_equal(rotx(math.pi/2),  R)
        nt.assert_array_almost_equal(rotx(math.pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(90, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(90, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotx(math.pi/2)), 1)
        
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(rotx(math.pi), R )
        nt.assert_array_almost_equal(rotx(math.pi, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(180, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotx(math.pi)), 1)
        
    
    def test_roty(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(roty(0),  R)
        nt.assert_array_almost_equal(roty(0, unit='rad'), R)
        nt.assert_array_almost_equal(roty(0, unit='deg'), R)
        nt.assert_array_almost_equal(roty(0, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(roty(0)), 1)
        
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        nt.assert_array_almost_equal(roty(math.pi/2),  R)
        nt.assert_array_almost_equal(roty(math.pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(roty(90, unit='deg'), R)
        nt.assert_array_almost_equal(roty(90, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(roty(math.pi/2)), 1)
        
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(roty(math.pi), R )
        nt.assert_array_almost_equal(roty(math.pi, unit='rad'), R)
        nt.assert_array_almost_equal(roty(180, unit='deg'), R)
        nt.assert_array_almost_equal(roty(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(roty(math.pi)), 1)
        
    
    def test_rotz(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(0),  R)
        nt.assert_array_almost_equal(rotz(0, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(0, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(0, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotz(0)), 1)
        
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(math.pi/2),  R)
        nt.assert_array_almost_equal(rotz(math.pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(90, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(90, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotz(math.pi/2)), 1)
        
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(math.pi), R )
        nt.assert_array_almost_equal(rotz(math.pi, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(180, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotz(math.pi)), 1)
        
    def test_trotX(self):
        T = np.array([[1, 0, 0, 3], [0, 0, -1, 4], [0, 1, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotx(math.pi/2, t=[3,4,5]),  T)
        nt.assert_array_almost_equal(trotx(math.pi/2, t=(3,4,5)),  T)
        nt.assert_array_almost_equal(trotx(math.pi/2, t=np.array([3,4,5])),  T)
        
        T = np.array([[0, 0, 1, 3], [0, 1, 0, 4], [-1, 0, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(troty(math.pi/2, t=[3,4,5]),  T)
        nt.assert_array_almost_equal(troty(math.pi/2, t=(3,4,5)),  T)
        nt.assert_array_almost_equal(troty(math.pi/2, t=np.array([3,4,5])),  T)
        
        T = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotz(math.pi/2, t=[3,4,5]),  T)
        nt.assert_array_almost_equal(trotz(math.pi/2, t=(3,4,5)),  T)
        nt.assert_array_almost_equal(trotz(math.pi/2, t=np.array([3,4,5])),  T)

 
    def test_Rt(self):
        nt.assert_array_almost_equal(rotx(0.3), t2r(trotx(0.3)))
        nt.assert_array_almost_equal(trotx(0.3), r2t(rotx(0.3)))
        
        R = rotx(0.2)
        t = [3, 4, 5]
        T = rt2tr(R,t)
        nt.assert_array_almost_equal(t2r(T), R)
        nt.assert_array_almost_equal(transl(T), np.array(t))
        
    def test_checks(self):
        
        # 3D case, with rotation matrix
        R = np.eye(3)
        nt.assert_equal( isR(R),          True )
        nt.assert_equal( isrot2(R),       False )
        nt.assert_equal( isrot(R),        True )
        nt.assert_equal( ishom(R),        False)
        nt.assert_equal( ishom2(R),       True )
        nt.assert_equal( isrot2(R, True), False )
        nt.assert_equal( isrot(R, True),  True )
        nt.assert_equal( ishom(R, True),  False )
        nt.assert_equal( ishom2(R, True), True )
        
        # 3D case, invalid rotation matrix
        R = np.eye(3)
        R[0,1] = 2
        nt.assert_equal( isR(R),          False )
        nt.assert_equal( isrot2(R),       False )
        nt.assert_equal( isrot(R),        True )
        nt.assert_equal( ishom(R),        False)
        nt.assert_equal( ishom2(R),       True )
        nt.assert_equal( isrot2(R, True), False )
        nt.assert_equal( isrot(R, True),  False )
        nt.assert_equal( ishom(R, True),  False )
        nt.assert_equal( ishom2(R, True), False )
        
        # 3D case, with rotation matrix
        T = np.array([[1, 0, 0, 3],[0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        nt.assert_equal( isR(T),          False )
        nt.assert_equal( isrot2(T),       False )
        nt.assert_equal( isrot(T),        False )
        nt.assert_equal( ishom(T),        True)
        nt.assert_equal( ishom2(T),       False )
        nt.assert_equal( isrot2(T, True), False )
        nt.assert_equal( isrot(T, True),  False )
        nt.assert_equal( ishom(T, True),  True )
        nt.assert_equal( ishom2(T, True), False )
        
        # 3D case, ivalid rotation matrix
        T = np.array([[1, 0, 0, 3],[0, 1, 1, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        nt.assert_equal( isR(T),          False )
        nt.assert_equal( isrot2(T),       False )
        nt.assert_equal( isrot(T),        False )
        nt.assert_equal( ishom(T),        True)
        nt.assert_equal( ishom2(T),       False )
        nt.assert_equal( isrot2(T, True), False )
        nt.assert_equal( isrot(T, True),  False )
        nt.assert_equal( ishom(T, True),  False )
        nt.assert_equal( ishom2(T, True), False )
        
    def test_rpy2r(self):
    
        r2d = 180/math.pi
        
        # default zyx order
        R = rotz(0.3) @ roty(0.2) @ rotx(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(rpy2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), R)
        nt.assert_array_almost_equal(rpy2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), R)
        
        # xyz order
        R = rotx(0.3) @ roty(0.2) @ rotz(0.1);
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='xyz'), R)
        
    
        # yxz order
        R = roty(0.3) @ rotx(0.2) @ rotz(0.1);
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='yxz'), R)
        
    def test_rpy2tr(self):
    
        r2d = 180/math.pi
        
        # default zyx order
        T = trotz(0.3) @ troty(0.2) @ trotx(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(rpy2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), T)
        
        # xyz order
        T = trotx(0.3) @ troty(0.2) @ trotz(0.1);
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='xyz'), T)
        
    
        # yxz order
        T = troty(0.3) @ trotx(0.2) @ trotz(0.1);
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='yxz'), T)

    def test_eul2r(self):
    
        r2d = 180/math.pi
        
        # default zyx order
        R = rotz(0.1) @ roty(0.2) @ rotz(0.3)
        nt.assert_array_almost_equal(eul2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(eul2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(eul2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), R)
        nt.assert_array_almost_equal(eul2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), R)
    
    def test_eul2tr(self):
    
        r2d = 180/math.pi
        
        # default zyx order
        T = trotz(0.1) @ troty(0.2) @ trotz(0.3)
        nt.assert_array_almost_equal(eul2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(eul2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(eul2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), T)
        nt.assert_array_almost_equal(eul2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), T)
        
class TestLie(unittest.TestCase):

    def test_vex(self):
        S = np.array([
            [ 0,    -3],
            [ 3,     0]
            ])
        
        nt.assert_array_almost_equal( vex(S), np.array([3]))
        nt.assert_array_almost_equal( vex(-S), np.array([-3]))
        
        S = np.array([
            [ 0,    -3,     2],
            [ 3,     0,    -1],
            [-2,     1,     0]
            ])
        
        nt.assert_array_almost_equal( vex(S), np.array([1, 2, 3]))
        nt.assert_array_almost_equal( vex(-S), -np.array([1,  2, 3]))
                  
    def test_skew(self):
        R = skew(3);
        nt.assert_equal( isrot2(R, check=False), True )  # check size
        nt.assert_array_almost_equal( np.linalg.norm(R.T+ R), 0) # check is skew
        nt.assert_array_almost_equal( vex(R), np.array([3])) # check contents, vex already verified
        
        R = skew([1, 2, 3]);
        nt.assert_equal( isrot(R, check=False), True )  # check size
        nt.assert_array_almost_equal( np.linalg.norm(R.T+ R), 0) # check is skew
        nt.assert_array_almost_equal( vex(R), np.array([1, 2, 3])) # check contents, vex already verified

    def test_vexa(self):
        
        S = np.array([
            [ 0,    -3,    1],
            [ 3,     0,    2],
            [ 0,     0,    0]
            ])
        nt.assert_array_almost_equal( vexa(S), np.array([1,2,3]))
        
        S = np.array([
            [ 0,     3,   -1],
            [-3,     0,    2],
            [ 0,     0,    0]
            ])
        nt.assert_array_almost_equal( vexa(S), np.array([-1,2,-3]))
        
        S = np.array([
            [ 0,    -6,     5,     1],
            [ 6,     0,    -4,     2],
            [-5,     4,     0,     3],
            [ 0,     0,     0,     0 ]
            ])
        nt.assert_array_almost_equal( vexa(S), np.array([1,2,3,4,5,6]))
    
        S = np.array([
            [ 0,    6,     5,     1],
            [-6,    0,     4,    -2],
            [-5,   -4,     0,     3],
            [ 0,    0,     0,     0 ]
            ]);
        nt.assert_array_almost_equal( vexa(S), np.array([1,-2,3,-4,5,-6]))

    def test_skewa(self):
        T = skewa([3, 4, 5])
        nt.assert_equal( ishom2(T, check=False), True )  # check size
        R = t2r(T)
        nt.assert_equal( np.linalg.norm(R.T+ R), 0)  # check is skew
        nt.assert_array_almost_equal( vexa(T), np.array([3, 4, 5])) # check contents, vexa already verified

        T = skewa([1, 2, 3, 4, 5, 6])
        nt.assert_equal( ishom(T, check=False), True )  # check size
        R = t2r(T)
        nt.assert_equal( np.linalg.norm(R.T+ R), 0)  # check is skew
        nt.assert_array_almost_equal( vexa(T), np.array([1, 2, 3, 4, 5, 6])) # check contents, vexa already verified
        
    def test_trlog(self):

    
        #%%% SO(3) tests
        # zero rotation case
        nt.assert_array_almost_equal(trlog( np.eye(3) ), skew([0, 0, 0]));
        
        # rotation by pi case
        nt.assert_array_almost_equal(trlog( rotx(pi) ), skew([math.pi, 0, 0]));
        nt.assert_array_almost_equal(trlog( roty(pi) ), skew([0, math.pi, 0]));
        nt.assert_array_almost_equal(trlog( rotz(pi) ), skew([0, 0, math.pi]));
        
        # general case
        nt.assert_array_almost_equal(trlog( rotx(0.2) ), skew([0.2, 0, 0]));
        nt.assert_array_almost_equal(trlog( roty(0.3) ), skew([0, 0.3, 0]));
        nt.assert_array_almost_equal(trlog( rotz(0.4) ), skew([0, 0, 0.4]));
        
        
        R = rotx(0.2) @ roty(0.3) @ rotz(0.4);
        [th,w] = trlog(R);
        nt.assert_array_almost_equal( logm(R), skew(th*w))
        
        #%% SE(3) tests
        
        # pure translation
        nt.assert_array_almost_equal(trlog( transl([1, 2, 3]) ), np.array([[0, 0, 0, 1], [ 0, 0, 0, 2], [ 0, 0, 0, 3], [ 0, 0, 0, 0]]))
        
        # mixture
        T = transl([1, 2, 3]) @ trotx(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T));
        
        T = transl([1, 2, 3]) @ troty(0.3);
        nt.assert_array_almost_equal(trlog(T), logm(T));
        
        [th,w] = trlog(T);
        nt.assert_array_almost_equal( logm(T), skewa(th*w))
        
    

    def test_trexp(self):
    
        #%% SO(3) tests
        
        #% so(3)
        
        # zero rotation case
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0])), np.eye(3,3));
        
        #% so(3), theta
        
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0]), 1), np.eye(3,3));
        
        # rotation by pi case
        nt.assert_array_almost_equal(trexp(skew([math.pi, 0, 0])), rotx(math.pi));
        nt.assert_array_almost_equal(trexp(skew([0, math.pi, 0])), roty(math.pi));
        nt.assert_array_almost_equal(trexp(skew([0, 0, math.pi])), rotz(math.pi));
        
        # general case
        nt.assert_array_almost_equal(trexp(skew([0.2, 0, 0])), rotx(0.2));
        nt.assert_array_almost_equal(trexp(skew([0, 0.3, 0])), roty(0.3));
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0.4])), rotz(0.4));
        
        nt.assert_array_almost_equal(trexp(skew([1, 0, 0]), 0.2), rotx(0.2));
        nt.assert_array_almost_equal(trexp(skew([0, 1, 0]), 0.3), roty(0.3));
        nt.assert_array_almost_equal(trexp(skew([0, 0, 1]), 0.4), rotz(0.4));
        
        nt.assert_array_almost_equal(trexp([1, 0, 0], 0.2), rotx(0.2));
        nt.assert_array_almost_equal(trexp([0, 1, 0], 0.3), roty(0.3));
        nt.assert_array_almost_equal(trexp([0, 0, 1], 0.4), rotz(0.4));
        
        nt.assert_array_almost_equal(trexp([1, 0, 0]*0.2), rotx(0.2));
        nt.assert_array_almost_equal(trexp([0, 1, 0]*0.3), roty(0.3));
        nt.assert_array_almost_equal(trexp([0, 0, 1]*0.4), rotz(0.4));
        
        
        #%% SE(3) tests
        
        #% sigma = se(3)
        # pure translation
        nt.assert_array_almost_equal(trexp( skewa([1, 2, 3, 0, 0, 0]) ), transl([1, 2, 3]));
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0.2, 0, 0]) ), trotx(0.2));
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 0.3, 0]) ), troty(0.3));
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 0, 0.4]) ), trotz(0.4));
        
        # mixture
        T = transl([1, 2, 3])*trotx(0.2)*troty(0.3)*trotz(0.4);
        nt.assert_array_almost_equal(trexp(logm(T)), T);
        
        #% twist vector
        nt.assert_array_almost_equal(trexp( double(Twist(T))), T);
        
        #% (sigma, theta)
        nt.assert_array_almost_equal(trexp( skewa([1, 0, 0, 0, 0, 0]), 2), transl([2, 0, 0]));
        nt.assert_array_almost_equal(trexp( skewa([0, 1, 0, 0, 0, 0]), 2), transl([0, 2, 0]));
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 1, 0, 0, 0]), 2), transl([0, 0, 2]));
        
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 1, 0, 0]), 0.2), trotx(0.2));
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 1, 0]), 0.2), troty(0.2));
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 0, 1]), 0.2), trotz(0.2));
        
        
        #% (twist, theta)
        nt.assert_array_almost_equal(trexp(Twist('R', [1, 0, 0], [0, 0, 0]).S, 0.3), trotx(0.3));
        
        
        T = transl([1, 2, 3])*troty(0.3);
        nt.assert_array_almost_equal(trexp(logm(T)), T);
        

  
        
    # test tr2rt rt2tr
    # trotX with t=
    







        

            
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    
    unittest.main()