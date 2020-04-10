#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:12:56 2020

@author: corkep
"""
import numpy as np
import math
import argcheck as check
import transforms as tr

def qone():
    return np.r_[1, 0,0,0]

def qpure(v):
    v = check.getvector(v,3)
    return np.r_[0, v]

def qnorm(q):
    q = check.getvector(q,4)
    return np.linalg.norm(q)
    
def qunit(q):
    q = check.getvector(q,4)
    nm = np.linalg.norm(q)
    assert abs(nm) >  10*np.finfo(np.float64).eps, 'cannot normalize (near) zero length quaternion'
    return q / nm

def qvec3(q):
    q = check.getvector(q,4)
    if q[0] >= 0:
        return q[1:4]
    else:
        return -q[1:4]

def qqmul(q1, q2):
    q1 = check.getvector(q1,4)
    q2 = check.getvector(q2,4)
    s1 = q1[0]; v1 = q1[1:4]
    s2 = q2[0]; v2 = q2[1:4]
    
    return np.r_[s1 * s2 - np.dot(v1, v2), s1 * v2 + s2 * v1 + np.cross(v1, v2)]

def qpow(q, power):
    q = check.getvector(q,4)
    assert type(power) is int, "Power must be an integer"
    qr = qone()
    for i in range(0, abs(power)):
        qr = qqmul(qr, q)

    if power < 0:
        qr = qconj(qr)

    return qr

def qconj(q):
    q = check.getvector(q,4)
    return np.r_[q[0], -q[1:4]]

def qvmul(q, v):
    pass

def q2r(q):
    q = check.getvector(q,4)
    s = q[0]; x = q[1]; y = q[2]; z = q[3]
    return np.array([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - s * z), 2 * (x * z + s * y)],
                          [2 * (x * y + s * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - s * x)],
                          [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x ** 2 + y ** 2)]])

def v32q(v):
    v = check.getvector(v,3)
    s = 1 - np.linalg.norm(v)
    return np.r_[s, v]

def r2q(R):
    assert tr.isrot(R, check=True), "Argument must be 3x3 rotation matrix"
    qs = math.sqrt(np.trace(R) + 1) / 2.0
    kx = R[2,1] - R[1,2]  # Oz - Ay
    ky = R[0,2] - R[2,0]  # Ax - Nz
    kz = R[1,0] - R[0,1]  # Ny - Ox

    if (R[0,0] >= R[1,1]) and (R[0,0] >= R[2,2]):
        kx1 = R[0,0] - R[1,1] - R[2,2] + 1  # Nx - Oy - Az + 1
        ky1 = R[1,0] + R[0,1]  # Ny + Ox
        kz1 = R[2,0] + R[0,2]  # Nz + Ax
        add = (kx >= 0)
    elif R[1,1] >= R[2,2]:
        kx1 = R[1,0] + R[0,1]  # Ny + Ox
        ky1 = R[1,1] - R[0,0] - R[2,2] + 1  # Oy - Nx - Az + 1
        kz1 = R[2,1] + R[1,2]  # Oz + Ay
        add = (ky >= 0)
    else:
        kx1 = R[2,0] + R[0,2]  # Nz + Ax
        ky1 = R[2,1] + R[1,2]  # Oz + Ay
        kz1 = R[2,2] - R[0,0] - R[1,1] + 1  # Az - Nx - Oy + 1
        add = (kz >= 0)

    if add:
        kx = kx + kx1
        ky = ky + ky1
        kz = kz + kz1
    else:
        kx = kx - kx1
        ky = ky - ky1
        kz = kz - kz1

    kv = np.r_[kx, ky, kz]
    nm = np.linalg.norm(kv)
    if abs(nm) <  100*np.finfo(np.float64).eps:
        return qone()
    else:
        return np.r_[qs, (math.sqrt(1.0 - qs ** 2) / nm) * kv]
    
def qslerp(q1, q2, s, shortest=False):
    assert 0 <= s <= 1, 's must be in the interval [0,1]'
    q1 = check.getvector(q1,4)
    q2 = check.getvector(q2,4)

    dot = np.dot(q1,q2)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if shortest:
        if dot < 0:
            q1 = - q1
            dot = -dot

    dot = np.clip(dot, -1, 1)  # Clip within domain of acos()
    theta_0 = math.acos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * s  # theta = angle between v0 and result
    s1 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0)
    s2 = math.sin(theta) / math.sin(theta_0)
    return (q1 * s1) + (q2 * s2)

def qmatrix(q):
    q = check.getvector(q,4)
    s = q[0]; x = q[1]; y = q[2]; z = q[3]
    return np.array([[s, -x, -y, -z],
                  [x, s, -z, y],
                  [y, z, s, -x],
                  [z, -y, x, s]])

def qequal(q1, q2, tol=100):
    q1 = check.getvector(q1,4)
    q2 = check.getvector(q2,4)
    tol *= 100*np.finfo(np.float64).eps
    return (np.sum(np.abs(q1 - q2)) < tol) or (np.sum(np.abs(q1 + q2)) < tol)
    
def qdot(q, w):
    E = q[0] * (np.eye(3, 3)) - tr.skew(q[1:4])
    return 0.5 * np.r_[-dot(q[1:4], w), E@omega]

def qdotb(q, w):
    E = q[0] * (np.eye(3, 3)) + tr.skew(q[1:4])
    return 0.5 * np.r_[-dot(q[1:4], w), E@omega]

def angle(q1, q2):

    """
    %UnitQuaternion.angle Angle between two UnitQuaternions
    %
    % A = Q1.angle(Q2) is the angle (in radians) between two UnitQuaternions Q1 and Q2.
    %
    % Notes::
    % - If either, or both, of Q1 or Q2 are vectors, then the result is a vector.
    %  - if Q1 is a vector (1xN) then A is a vector (1xN) such that A(i) = P1(i).angle(Q2).
    %  - if Q2 is a vector (1xN) then A is a vector (1xN) such that A(i) = P1.angle(P2(i)).
    %  - if both Q1 and Q2 are vectors (1xN) then A is a vector (1xN) such 
    %    that A(i) = P1(i).angle(Q2(i)).
    %
    % References::
    % - Metrics for 3D rotations: comparison and analysis, Du Q. Huynh,
    %   J.Math Imaging Vis. DOFI 10.1007/s10851-009-0161-2.
    %
    % See also Quaternion.angvec.
    """
    # TODO different methods
    
    q1 = check.getvector(q1,4)
    q2 = check.getvector(q2,4)
    return 2.0*math.atan2( qnorm(q1-q2), qnorm(q1+q2))
            
            
def qprint(q, delim=('<', '>'), fmt='%f', file=None):
    q = check.getvector(q,4)
    s = "%f %s%f, %f, %f %s" % (q[0], delim[0], q[1], q[2], q[2], delim[1])
    if file:
        print(s, file=file)
    return s
    
if __name__ == '__main__':

    import numpy.testing as nt
    import unittest
    
    from transforms import *
            
    class TestQuaternion(unittest.TestCase):
        def test_ops(self):
            nt.assert_array_almost_equal(qone(), np.r_[1,0,0,0])
    
            nt.assert_array_almost_equal(qpure(np.r_[1,2,3]), np.r_[0,1,2,3])
            
            nt.assert_equal(qnorm(np.r_[1,2,3,4]), math.sqrt(30))
                            
            nt.assert_array_almost_equal(qunit(np.r_[1,2,3,4]), np.r_[1,2,3,4]/math.sqrt(30))
            
            nt.assert_array_almost_equal(qqmul(np.r_[1,2,3,4],np.r_[5,6,7,8]), np.r_[-60,12,30,24])
            nt.assert_array_almost_equal(qqmul(np.r_[1,2,3,4],np.r_[1,2,3,4]), np.r_[-28,4,6,8])
            
            nt.assert_array_almost_equal(qmatrix(np.r_[1,2,3,4])@np.r_[5,6,7,8], np.r_[-60,12,30,24])
            nt.assert_array_almost_equal(qmatrix(np.r_[1,2,3,4])@np.r_[1,2,3,4], np.r_[-28,4,6,8])
            
            nt.assert_array_almost_equal(qpow(np.r_[1,2,3,4],0), np.r_[1,0,0,0])
            nt.assert_array_almost_equal(qpow(np.r_[1,2,3,4],1), np.r_[1,2,3,4])
            nt.assert_array_almost_equal(qpow(np.r_[1,2,3,4],2), np.r_[-28,4,6,8])
            nt.assert_array_almost_equal(qpow(np.r_[1,2,3,4],-1), np.r_[1,-2,-3,-4])
            nt.assert_array_almost_equal(qpow(np.r_[1,2,3,4],-2), np.r_[-28,-4,-6,-8])
            
            nt.assert_equal(qequal(np.r_[1,2,3,4], np.r_[1,2,3,4]), True)
            nt.assert_equal(qequal(np.r_[1,2,3,4], np.r_[5,6,7,8]), False)
            
            s = qprint
            
        def test_rotation(self):
            nt.assert_array_almost_equal(r2q(tr.rotx(180,'deg')), np.r_[0,1,0,0])
            nt.assert_array_almost_equal(r2q(tr.roty(180,'deg')), np.r_[0,0,1,0])
            nt.assert_array_almost_equal(r2q(tr.rotz(180,'deg')), np.r_[0,0,0,1])
            
            nt.assert_array_almost_equal(q2r(np.r_[0,1,0,0]), tr.rotx(180,'deg'))
            nt.assert_array_almost_equal(q2r(np.r_[0,0,1,0]), tr.roty(180,'deg'))
            nt.assert_array_almost_equal(q2r(np.r_[0,0,0,1]), tr.rotz(180,'deg'))
    
        def test_slerp(self):
            q1 = np.r_[0,1,0,0]
            q2 = np.r_[0,0,1,0]
            
            nt.assert_array_almost_equal(qslerp(q1, q2, 0), q1)
            nt.assert_array_almost_equal(qslerp(q1, q2, 1), q2)
            nt.assert_array_almost_equal(qslerp(q1, q2, 0.5), np.r_[0,1,1,0]/math.sqrt(2))
        
        def test_rotx(self):
             pass
        
    unittest.main()