#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:19:04 2020

@author: corkep
"""

# Some unit tests

import numpy.testing as nt
import unittest
from math import pi

from spatialmath.base.transforms import *

class TestVector(unittest.TestCase):
    
    def test_unit(self):
        
        nt.assert_array_almost_equal(unit([1,0,0]),  np.r_[1,0,0])
        nt.assert_array_almost_equal(unit([0,1,0]),  np.r_[0,1,0])
        nt.assert_array_almost_equal(unit([0,0,1]),  np.r_[0,0,1])
        
        nt.assert_array_almost_equal(unit((1,0,0)),  np.r_[1,0,0])
        nt.assert_array_almost_equal(unit((0,1,0)),  np.r_[0,1,0])
        nt.assert_array_almost_equal(unit((0,0,1)),  np.r_[0,0,1])
        
        nt.assert_array_almost_equal(unit(np.r_[1,0,0]),  np.r_[1,0,0])
        nt.assert_array_almost_equal(unit(np.r_[0,1,0]),  np.r_[0,1,0])
        nt.assert_array_almost_equal(unit(np.r_[0,0,1]),  np.r_[0,0,1])
        
        nt.assert_array_almost_equal(unit([9,0,0]),  np.r_[1,0,0])
        nt.assert_array_almost_equal(unit([0,9,0]),  np.r_[0,1,0])
        nt.assert_array_almost_equal(unit([0,0,9]),  np.r_[0,0,1])
        
    def test_isunitvec(self):
        nt.assert_equal(isunitvec([1,0,0]), True)
        nt.assert_equal(isunitvec((1,0,0)), True)
        nt.assert_equal(isunitvec(np.r_[1,0,0]), True)
        
        nt.assert_equal(isunitvec([9,0,0]), False)
        nt.assert_equal(isunitvec((9,0,0)), False)
        nt.assert_equal(isunitvec(np.r_[9,0,0]), False)
        
    def test_norm(self):
        nt.assert_array_almost_equal(norm([0,0,0]),  0)
        nt.assert_array_almost_equal(norm([1,2,3]),  math.sqrt(14))
        
    def test_isunittwist(self):
        # unit rotational twist
        nt.assert_equal(isunittwist([1,2,3, 1,0,0]), True)
        nt.assert_equal(isunittwist((1,2,3, 1,0,0)), True)
        nt.assert_equal(isunittwist(np.r_[1,2,3, 1,0,0]), True)
        
        # not a unit rotational twist
        nt.assert_equal(isunittwist([1,2,3, 1,0,1]), False)

        # unit translation twist
        nt.assert_equal(isunittwist([1,0,0, 0,0,0]), True)

        # not a unit translation twist
        nt.assert_equal(isunittwist([2,0,0, 0,0,0]), False)
        
    def test_iszerovec(self):
        nt.assert_equal(iszerovec([0]), True)
        nt.assert_equal(iszerovec([0,0]), True)
        nt.assert_equal(iszerovec([0,0,0]), True)
        
        nt.assert_equal(iszerovec([1]), False)
        nt.assert_equal(iszerovec([0,1]), False)
        nt.assert_equal(iszerovec([0,1,0]), False)
        
class TestND(unittest.TestCase):
    def test_iseye(self):
        nt.assert_equal(iseye(np.eye(1)), True)
        nt.assert_equal(iseye(np.eye(2)), True)
        nt.assert_equal(iseye(np.eye(3)), True)
        nt.assert_equal(iseye(np.eye(5)), True)
        
        nt.assert_equal(iseye(2*np.eye(3)), False)
        nt.assert_equal(iseye(-np.eye(3)), False)
        nt.assert_equal(iseye( np.array([[1,0,0],[0,1,0]]) ), False)
        nt.assert_equal(iseye( np.array([1,0,0]) ), False)
    
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
        
        # 3D case, invalid rotation matrix
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
        
        # 3D case, invalid bottom row
        T = np.array([[1, 0, 0, 3],[0, 1, 1, 4], [0, 0, 1, 5], [9, 0, 0, 1]])
        nt.assert_equal( isR(T),          False )
        nt.assert_equal( isrot2(T),       False )
        nt.assert_equal( isrot(T),        False )
        nt.assert_equal( ishom(T),        True)
        nt.assert_equal( ishom2(T),       False )
        nt.assert_equal( isrot2(T, True), False )
        nt.assert_equal( isrot(T, True),  False )
        nt.assert_equal( ishom(T, True),  False )
        nt.assert_equal( ishom2(T, True), False )
        
        # skew matrices
        S = np.array([ 
            [ 0, 2], 
            [-2, 0] ])
        nt.assert_equal( isskew(S), True )
        S[0,0] = 1
        nt.assert_equal( isskew(S), False )
        
        S = np.array([
            [ 0, -3,  2],
            [ 3,  0, -1],
            [-2,  1,  0]])
        nt.assert_equal( isskew(S), True )
        S[0,0] = 1
        nt.assert_equal( isskew(S), False )
        
        # augmented skew matrices
        S = np.array([ 
            [ 0, 2, 3], 
            [-2, 0, 4],
            [ 0, 0, 0]])
        nt.assert_equal( isskewa(S), True )
        S[0,0] = 1
        nt.assert_equal( isskew(S), False )
        S[0,0] = 0
        S[2,0] = 1
        nt.assert_equal( isskew(S), False )
    
class Test2D(unittest.TestCase):
    def test_rot2(self):
        R = np.array([[1, 0], [0, 1]])
        nt.assert_array_almost_equal(rot2(0),  R)
        nt.assert_array_almost_equal(rot2(0, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(0, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(0, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(0)), 1)
        
        R = np.array([[0, -1], [1, 0]])
        nt.assert_array_almost_equal(rot2(pi/2),  R)
        nt.assert_array_almost_equal(rot2(pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(90, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(90, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(pi/2)), 1)
        
        R = np.array([[-1, 0], [0, -1]])
        nt.assert_array_almost_equal(rot2(pi), R)
        nt.assert_array_almost_equal(rot2(pi, unit='rad'), R)
        nt.assert_array_almost_equal(rot2(180, unit='deg'), R)
        nt.assert_array_almost_equal(rot2(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rot2(pi)), 1)
        
    def test_trot2(self):
        nt.assert_array_almost_equal(trot2(pi/2, t=[3,4]),  np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
        nt.assert_array_almost_equal(trot2(pi/2, t=(3,4)),  np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
        nt.assert_array_almost_equal(trot2(pi/2, t=np.array([3,4])),  np.array([[0, -1, 3], [1, 0, 4], [0, 0, 1]]))
    
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
        
        # 2D case, invalid rotation matrix
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
        
                
        # 2D case, invalid bottom row
        T = np.array([[1,1, 3],[0, 1, 4], [9, 0, 1]])
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
        nt.assert_array_almost_equal(rotx(pi/2),  R)
        nt.assert_array_almost_equal(rotx(pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(90, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(90, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotx(pi/2)), 1)
        
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(rotx(pi), R )
        nt.assert_array_almost_equal(rotx(pi, unit='rad'), R)
        nt.assert_array_almost_equal(rotx(180, unit='deg'), R)
        nt.assert_array_almost_equal(rotx(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotx(pi)), 1)
        
    
    def test_roty(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(roty(0),  R)
        nt.assert_array_almost_equal(roty(0, unit='rad'), R)
        nt.assert_array_almost_equal(roty(0, unit='deg'), R)
        nt.assert_array_almost_equal(roty(0, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(roty(0)), 1)
        
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        nt.assert_array_almost_equal(roty(pi/2),  R)
        nt.assert_array_almost_equal(roty(pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(roty(90, unit='deg'), R)
        nt.assert_array_almost_equal(roty(90, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(roty(pi/2)), 1)
        
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        nt.assert_array_almost_equal(roty(pi), R )
        nt.assert_array_almost_equal(roty(pi, unit='rad'), R)
        nt.assert_array_almost_equal(roty(180, unit='deg'), R)
        nt.assert_array_almost_equal(roty(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(roty(pi)), 1)
        
    
    def test_rotz(self):
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(0),  R)
        nt.assert_array_almost_equal(rotz(0, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(0, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(0, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotz(0)), 1)
        
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(pi/2),  R)
        nt.assert_array_almost_equal(rotz(pi/2, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(90, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(90, 'deg'), R )
        nt.assert_almost_equal(np.linalg.det(rotz(pi/2)), 1)
        
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        nt.assert_array_almost_equal(rotz(pi), R )
        nt.assert_array_almost_equal(rotz(pi, unit='rad'), R)
        nt.assert_array_almost_equal(rotz(180, unit='deg'), R)
        nt.assert_array_almost_equal(rotz(180, 'deg'), R)
        nt.assert_almost_equal(np.linalg.det(rotz(pi)), 1)
        
    def test_trotX(self):
        T = np.array([[1, 0, 0, 3], [0, 0, -1, 4], [0, 1, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotx(pi/2, t=[3,4,5]),  T)
        nt.assert_array_almost_equal(trotx(pi/2, t=(3,4,5)),  T)
        nt.assert_array_almost_equal(trotx(pi/2, t=np.array([3,4,5])),  T)
        
        T = np.array([[0, 0, 1, 3], [0, 1, 0, 4], [-1, 0, 0, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(troty(pi/2, t=[3,4,5]),  T)
        nt.assert_array_almost_equal(troty(pi/2, t=(3,4,5)),  T)
        nt.assert_array_almost_equal(troty(pi/2, t=np.array([3,4,5])),  T)
        
        T = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])
        nt.assert_array_almost_equal(trotz(pi/2, t=[3,4,5]),  T)
        nt.assert_array_almost_equal(trotz(pi/2, t=(3,4,5)),  T)
        nt.assert_array_almost_equal(trotz(pi/2, t=np.array([3,4,5])),  T)

 

        
    def test_rpy2r(self):
    
        r2d = 180/pi
        
        # default zyx order
        R = rotz(0.3) @ roty(0.2) @ rotx(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(rpy2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), R)
        nt.assert_array_almost_equal(rpy2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), R)
        
        # xyz order
        R = rotx(0.3) @ roty(0.2) @ rotz(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='xyz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='xyz'), R)
        
    
        # yxz order
        R = roty(0.3) @ rotx(0.2) @ rotz(0.1)
        nt.assert_array_almost_equal(rpy2r(0.1, 0.2, 0.3, order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1, 0.2, 0.3], order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='yxz'), R)
        nt.assert_array_almost_equal(rpy2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='yxz'), R)
        
    def test_rpy2tr(self):
    
        r2d = 180/pi
        
        # default zyx order
        T = trotz(0.3) @ troty(0.2) @ trotx(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(rpy2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), T)
        
        # xyz order
        T = trotx(0.3) @ troty(0.2) @ trotz(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='xyz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='xyz'), T)
        
    
        # yxz order
        T = troty(0.3) @ trotx(0.2) @ trotz(0.1)
        nt.assert_array_almost_equal(rpy2tr(0.1, 0.2, 0.3, order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1, 0.2, 0.3], order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg', order='yxz'), T)
        nt.assert_array_almost_equal(rpy2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg', order='yxz'), T)

    def test_eul2r(self):
    
        r2d = 180/pi
        
        # default zyx order
        R = rotz(0.1) @ roty(0.2) @ rotz(0.3)
        nt.assert_array_almost_equal(eul2r(0.1, 0.2, 0.3), R)
        nt.assert_array_almost_equal(eul2r([0.1, 0.2, 0.3]), R)
        nt.assert_array_almost_equal(eul2r(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), R)
        nt.assert_array_almost_equal(eul2r([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), R)
    
    def test_eul2tr(self):
    
        r2d = 180/pi
        
        # default zyx order
        T = trotz(0.1) @ troty(0.2) @ trotz(0.3)
        nt.assert_array_almost_equal(eul2tr(0.1, 0.2, 0.3), T)
        nt.assert_array_almost_equal(eul2tr([0.1, 0.2, 0.3]), T)
        nt.assert_array_almost_equal(eul2tr(0.1*r2d, 0.2*r2d, 0.3*r2d, unit='deg'), T)
        nt.assert_array_almost_equal(eul2tr([0.1*r2d, 0.2*r2d, 0.3*r2d], unit='deg'), T)
        
    def test_tr2rpy(self):
        rpy = np.r_[0.1, 0.2, 0.3]
        R = rpy2r(rpy)
        nt.assert_array_almost_equal(tr2rpy(R), rpy)
        nt.assert_array_almost_equal(tr2rpy(R, unit='deg'), rpy*180/pi)
        
        T = rpy2tr(rpy)
        nt.assert_array_almost_equal(tr2rpy(T), rpy,)
        nt.assert_array_almost_equal(tr2rpy(T, unit='deg'), rpy*180/pi)
        
        # xyz order
        R = rpy2r(rpy, order='xyz')
        nt.assert_array_almost_equal(tr2rpy(R, order='xyz'), rpy)
        nt.assert_array_almost_equal(tr2rpy(R, unit='deg', order='xyz'), rpy*180/pi)
        
        
        T = rpy2tr(rpy, order='xyz')
        nt.assert_array_almost_equal(tr2rpy(T, order='xyz'), rpy)
        nt.assert_array_almost_equal(tr2rpy(T, unit='deg', order='xyz'), rpy*180/pi)
        
        
        # corner cases
        seq = 'zyx'
        ang = [pi, 0 ,0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi/2, 0]; # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi/2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        
        seq = 'xyz'
        ang = [pi, 0 ,0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi/2, 0]; # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi/2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        
        seq = 'yxz'
        ang = [pi, 0 ,0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, 0, pi]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, pi/2, 0]; # singularity
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        ang = [0, -pi/2, 0]
        a = rpy2tr(ang, order=seq)
        nt.assert_array_almost_equal(rpy2tr(tr2rpy(a, order=seq), order=seq), a)
        
    
    def test_tr2eul(self):
    
        eul = np.r_[0.1, 0.2, 0.3]
        R = eul2r(eul)
        nt.assert_array_almost_equal(tr2eul(R), eul)
        nt.assert_array_almost_equal(tr2eul(R, unit='deg'), eul*180/pi)
        
        T = eul2tr(eul)
        nt.assert_array_almost_equal(tr2eul(T), eul)
        nt.assert_array_almost_equal(tr2eul(T, unit='deg'), eul*180/pi)

        # test singularity case
        eul = [0.1, 0, 0.3]
        R = eul2r(eul)
        nt.assert_array_almost_equal(eul2r( tr2eul(R) ), R)
        nt.assert_array_almost_equal(eul2r( tr2eul(R, unit='deg'), unit='deg'), R)
    
        # test flip
        eul = [-0.1, 0.2, 0.3]
        R = eul2r(eul)
        eul2 = tr2eul(R, flip=True)
        nt.assert_equal(eul2[0] > 0, True)
        nt.assert_array_almost_equal(eul2r(eul2), R)
    
    def test_tr2angvec(self):
    
        # null rotation
        # - vector isn't defined here, but RTB sets it (0 0 0)
        [theta, v] = tr2angvec(np.eye(3,3))
        nt.assert_array_almost_equal(theta, 0.0)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 0])
        
        tr2angvec(eye(3,3))
        
        # canonic rotations
        [theta, v] = tr2angvec(rotx(pi/2))
        nt.assert_array_almost_equal(theta, pi/2)
        nt.assert_array_almost_equal(v, np.r_[1, 0, 0])
        
        [theta, v] = tr2angvec(roty(pi/2))
        nt.assert_array_almost_equal(theta, pi/2)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])
        
        [theta, v] = tr2angvec(rotz(pi/2))
        nt.assert_array_almost_equal(theta, pi/2)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 1])
        
        # null rotation
        [theta, v] = tr2angvec(eye(4,4))
        nt.assert_array_almost_equal(theta, 0.0)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 0])
        
        # canonic rotations
        [theta, v] = tr2angvec(trotx(pi/2))
        nt.assert_array_almost_equal(theta, pi/2)
        nt.assert_array_almost_equal(v, np.r_[1, 0, 0])
        
        [theta, v] = tr2angvec(troty(pi/2))
        nt.assert_array_almost_equal(theta, pi/2)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])
        
        [theta, v] = tr2angvec(trotz(pi/2))
        nt.assert_array_almost_equal(theta, pi/2)
        nt.assert_array_almost_equal(v, np.r_[0, 0, 1])
        
        [theta, v] = tr2angvec(roty(pi/2), unit='deg')
        nt.assert_array_almost_equal(theta, 90)
        nt.assert_array_almost_equal(v, np.r_[0, 1, 0])
        
        R = cat(3, rotx(pi/2), roty(pi/2), rotz(pi/2))
        [theta, v] = tr2angvec(R)
        nt.assert_array_almost_equal(theta, pi/2*np.r_[1, 1, 1])
        nt.assert_array_almost_equal(v, eye(3,3))
        
        T = cat(3, trotx(pi/2), troty(pi/2), trotz(pi/2))
        [theta, v] = tr2angvec(T)
        nt.assert_array_almost_equal(theta, pi/2*np.r_[1, 1, 1])
        nt.assert_array_almost_equal(v, eye(3,3))
        
            
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
        R = skew(3)
        nt.assert_equal( isrot2(R, check=False), True )  # check size
        nt.assert_array_almost_equal( np.linalg.norm(R.T+ R), 0) # check is skew
        nt.assert_array_almost_equal( vex(R), np.array([3])) # check contents, vex already verified
        
        R = skew([1, 2, 3])
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
            ])
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
        nt.assert_array_almost_equal(trlog( np.eye(3) ), skew([0, 0, 0]))
        
        # rotation by pi case
        nt.assert_array_almost_equal(trlog( rotx(pi) ), skew([pi, 0, 0]))
        nt.assert_array_almost_equal(trlog( roty(pi) ), skew([0, pi, 0]))
        nt.assert_array_almost_equal(trlog( rotz(pi) ), skew([0, 0, pi]))
        
        # general case
        nt.assert_array_almost_equal(trlog( rotx(0.2) ), skew([0.2, 0, 0]))
        nt.assert_array_almost_equal(trlog( roty(0.3) ), skew([0, 0.3, 0]))
        nt.assert_array_almost_equal(trlog( rotz(0.4) ), skew([0, 0, 0.4]))
        
        
        R = rotx(0.2) @ roty(0.3) @ rotz(0.4)
        [th,w] = trlog(R)
        nt.assert_array_almost_equal( logm(R), skew(th*w))
        
        #%% SE(3) tests
        
        # pure translation
        nt.assert_array_almost_equal(trlog( transl([1, 2, 3]) ), np.array([[0, 0, 0, 1], [ 0, 0, 0, 2], [ 0, 0, 0, 3], [ 0, 0, 0, 0]]))
        
        # mixture
        T = transl([1, 2, 3]) @ trotx(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T))
        
        T = transl([1, 2, 3]) @ troty(0.3)
        nt.assert_array_almost_equal(trlog(T), logm(T))
        
        [th,w] = trlog(T)
        nt.assert_array_almost_equal( logm(T), skewa(th*w))
        
    

    def test_trexp(self):
    
        #%% SO(3) tests
        
        #% so(3)
        
        # zero rotation case
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0])), np.eye(3,3))
        
        #% so(3), theta
                
        # rotation by pi case
        nt.assert_array_almost_equal(trexp(skew([pi, 0, 0])), rotx(pi))
        nt.assert_array_almost_equal(trexp(skew([0, pi, 0])), roty(pi))
        nt.assert_array_almost_equal(trexp(skew([0, 0, pi])), rotz(pi))
        
        # general case
        nt.assert_array_almost_equal(trexp(skew([0.2, 0, 0])), rotx(0.2))
        nt.assert_array_almost_equal(trexp(skew([0, 0.3, 0])), roty(0.3))
        nt.assert_array_almost_equal(trexp(skew([0, 0, 0.4])), rotz(0.4))
        
        nt.assert_array_almost_equal(trexp(skew([1, 0, 0]), 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp(skew([0, 1, 0]), 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp(skew([0, 0, 1]), 0.4), rotz(0.4))
        
        nt.assert_array_almost_equal(trexp([1, 0, 0], 0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp([0, 1, 0], 0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp([0, 0, 1], 0.4), rotz(0.4))
        
        nt.assert_array_almost_equal(trexp(np.r_[1, 0, 0]*0.2), rotx(0.2))
        nt.assert_array_almost_equal(trexp(np.r_[0, 1, 0]*0.3), roty(0.3))
        nt.assert_array_almost_equal(trexp(np.r_[0, 0, 1]*0.4), rotz(0.4))
        
        
        #%% SE(3) tests
        
        #% sigma = se(3)
        # pure translation
        nt.assert_array_almost_equal(trexp( skewa([1, 2, 3, 0, 0, 0]) ), transl([1, 2, 3]))
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0.2, 0, 0]) ), trotx(0.2))
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 0.3, 0]) ), troty(0.3))
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 0, 0.4]) ), trotz(0.4))
        
        # mixture
        S = skewa([1,2,3, 0.1, -0.2, 0.3])
        nt.assert_array_almost_equal(trexp(S), expm(S))
        
        # twist vector
        #nt.assert_array_almost_equal(trexp( double(Twist(T))), T)
        
        # (sigma, theta)
        nt.assert_array_almost_equal(trexp( skewa([1, 0, 0, 0, 0, 0]), 2), transl([2, 0, 0]))
        nt.assert_array_almost_equal(trexp( skewa([0, 1, 0, 0, 0, 0]), 2), transl([0, 2, 0]))
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 1, 0, 0, 0]), 2), transl([0, 0, 2]))
        
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 1, 0, 0]), 0.2), trotx(0.2))
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 1, 0]), 0.2), troty(0.2))
        nt.assert_array_almost_equal(trexp( skewa([0, 0, 0, 0, 0, 1]), 0.2), trotz(0.2))
        
        
        # (twist, theta)
        #nt.assert_array_almost_equal(trexp(Twist('R', [1, 0, 0], [0, 0, 0]).S, 0.3), trotx(0.3))
        
        
        T = transl([1, 2, 3])*trotz(0.3)
        nt.assert_array_almost_equal(trexp(np.linalg.logm(T)), T)
        

    # test tr2rt rt2tr
    # trotX with t=
    







        

            
# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    
    unittest.main()
