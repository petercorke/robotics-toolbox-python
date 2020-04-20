# Created by: Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan
# 1 June, 2017
""" 
This modules contains functions to create and transform rotation matrices
and homogeneous tranformation matrices.
"""

import sys
import math
import numpy as np
from scipy.linalg import expm
import spatialmath.base.argcheck as argcheck

try:
    import vtk
except:
    print("VTK not installed\n")

try:
    print('Using SymPy')
    import sympy as sym
    def issymbol(x):
        return isinstance(x, sym.Symbol)
except:
    def issymbol(x):
        return False


def colvec(v):
    return np.array(v).reshape((len(v), 1))

# ---------------------------------------------------------------------------------------#
    
def _cos(theta):
    if issymbol(theta):
        return sym.cos(theta)
    else:
        return math.cos(theta)
        
def _sin(theta):
    if issymbol(theta):
        return sym.sin(theta)
    else:
        return math.sin(theta)

def rotx(theta, unit="rad"):
    """
    Create SO(3) rotation about X-axis

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``rotx(THETA)`` is an SO(3) rotation matrix (3x3) representing a rotation
      of THETA radians about the x-axis
    - ``rotx(THETA, "deg")`` as above but THETA is in degrees
    """

    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [1, 0, 0], 
            [0, ct, -st], 
            [0, st, ct]  ])
    if not isinstance(theta, sym.Symbol):
        R = R.round(15)
    return R


# ---------------------------------------------------------------------------------------#
def roty(theta, unit="rad"):
    """
    Create SO(3) rotation about Y-axis

    :param theta: rotation angle about Y-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``roty(THETA)`` is an SO(3) rotation matrix (3x3) representing a rotation
      of THETA radians about the y-axis
    - ``roty(THETA, "deg")`` as above but THETA is in degrees
    """

    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [ct, 0, st], 
            [0, 1, 0], 
            [-st, 0, ct]  ])
    if not isinstance(theta, sym.Symbol):
        R = R.round(15)
    return R


# ---------------------------------------------------------------------------------------#
def rotz(theta, unit="rad"):
    """
    Create SO(3) rotation about Z-axis

    :param theta: rotation angle about Z-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``rotz(THETA)`` is an SO(3) rotation matrix (3x3) representing a rotation
      of THETA radians about the z-axis
    - ``rotz(THETA, "deg")`` as above but THETA is in degrees
    """
    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [ct, -st, 0], 
            [st,  ct, 0], 
            [0,   0,  1]  ])
    if not isinstance(theta, sym.Symbol):
        R = R.round(15)
    return R


# ---------------------------------------------------------------------------------------#
def trotx(theta, unit="rad", t=None):
    """
    Create SE(3) pure rotation about X-axis

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: the translation, defaults to [0,0,0]
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    - ``trotx(THETA)`` is a homogeneous transformation (4x4) representing a rotation
      of THETA radians about the x-axis.
    - ``trotx(THETA, 'deg')`` as above but THETA is in degrees
    - ``trotx(THETA, 'rad', t=[x,y,z])`` as above with translation of [x,y,z]
    """
    T  = np.pad( rotx(theta, unit), (0,1) )
    if t is not None:
        T[:3,3] = argcheck.getvector(t, 3, 'array')
    T[3,3] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def troty(theta, unit="rad", t=None):
    """
    Create SE(3) pure rotation about Y-axis

    :param theta: rotation angle about Y-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: the translation, defaults to [0,0,0]
    :return: 4x4 homogeneous transformation matrix as a numpy array
    :rtype: numpy.ndarray, shape=(4,4)

    - ``troty(THETA)`` is a homogeneous transformation (4x4) representing a rotation
      of THETA radians about the y-axis.
    - ``troty(THETA, 'deg')`` as above but THETA is in degrees
    - ``troty(THETA, 'rad', t=[x,y,z])`` as above with translation of [x,y,z]
    """
    T  = np.pad( roty(theta, unit), (0,1) )
    if t is not None:
        T[:3,3] = argcheck.getvector(t, 3, 'array')
    T[3,3] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def trotz(theta, unit="rad", t=None):
    """
    Create SE(3) pure rotation about Z-axis

    :param theta: rotation angle about Z-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: the translation, defaults to [0,0,0]
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    - ``trotz(THETA)`` is a homogeneous transformation (4x4) representing a rotation
      of THETA radians about the z-axis.
    - ``trotz(THETA, 'deg')`` as above but THETA is in degrees
    - ``trotz(THETA, 'rad', t=[x,y,z])`` as above with translation of [x,y,z]
    """
    T  = np.pad( rotz(theta, unit), (0,1) )
    if t is not None:
        T[:3,3] = argcheck.getvector(t, 3, 'array')
    T[3,3] = 1.0
    return T

# ---------------------------------------------------------------------------------------#
def transl(x, y=None, z=None):
    """
    Create SE(3) pure translation, or extract translation from SE(3) matrix

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :param z: translation along Z-axis
    :type z: float
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    Create a translational SE(3) matrix:

    - ``T = transl( X, Y, Z )`` is an SE(3) homogeneous transform (4x4) representing a
      pure translation of X, Y and Z.
    - ``T = transl( V )`` as above but the translation is given by a 3-element
      list, dict, or a numpy array, row or column vector.


    Extract the translational part of an SE(3) matrix:

    - ``P = TRANSL(T)`` is the translational part of a homogeneous transform T as a
      3-element numpy array.
   """

    if np.isscalar(x):
        T = np.identity(4)
        T[:3,3] = [x, y, z]
        return T
    elif argcheck.isvector(x, 3):
        T = np.identity(4)
        T[:3,3] = argcheck.getvector(x, 3, out='array')
        return T
    elif argcheck.ismatrix(x, (4,4)):
        return x[:3,3]
    else:
        ValueError('bad argument')
        

# ---------------------------------------------------------------------------------------#
def rot2(theta, unit='rad'):
    """
    Create SO(2) rotation

    :param theta: rotation angle
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 2x2 rotation matrix
    :rtype: numpy.ndarray, shape=(2,2)

    - ``ROT2(THETA)`` is an SO(2) rotation matrix (2x2) representing a rotation of THETA radians.
    - ``ROT2(THETA, 'deg')`` as above but THETA is in degrees.
    """
    theta = argcheck.getunit(theta, unit)
    ct = _cos(theta)
    st = _sin(theta)
    R = np.array([
            [ct, -st], 
            [st, ct]  ])
    if not isinstance(theta, sym.Symbol):
        R = R.round(15)
    return R


# ---------------------------------------------------------------------------------------#
def trot2(theta, unit='rad', t=None):
    """
    Create SE(2) pure rotation 

    :param theta: rotation angle about X-axis
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param t: the translation, defaults to [0,0]
    :return: 3x3 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    - ``TROT2(THETA)`` is a homogeneous transformation (3x3) representing a rotation of
      THETA radians.
    - ``TROT2(THETA, 'deg')`` as above but THETA is in degrees.
    
    Notes:
    - Translational component is zero.
    """
    T  = np.pad( rot2(theta, unit), (0,1) )
    if t is not None:
        T[:2,2] = argcheck.getvector(t, 2, 'array')
    T[2,2] = 1.0
    return T


# ---------------------------------------------------------------------------------------#
def transl2(x, y=None):
    """
    Create SE(2) pure translation, or extract translation from SE(2) matrix

    :param x: translation along X-axis
    :type x: float
    :param y: translation along Y-axis
    :type y: float
    :return: homogeneous transform matrix or the translation elements of a homogeneous transform
    :rtype: numpy.ndarray, shape=(3,3)

    Create a translational SE(2) matrix:

    - ``T = transl2([X, Y])`` is an SE(2) homogeneous transform (3x3) representing a
      pure translation.
    - ``T = transl2( V )`` as above but the translation is given by a 2-element
      list, dict, or a numpy array, row or column vector.


    Extract the translational part of an SE(2) matrix:

    P = TRANSL2(T) is the translational part of a homogeneous transform as a
    2-element numpy array.  
    """

    if np.isscalar(x):
        T = np.identity(3)
        T[:2,2] = [x, y]
        return T
    elif argcheck.isvector(x, 2):
        T = np.identity(3)
        T[:2,2] = argcheck.getvector(x, 2)
        return T
    elif argcheck.ismatrix(x, (3,3)):
        return x[:2,2]
    else:
        ValueError('bad argument')



# ---------------------------------------------------------------------------------------#
def unit(v):
    """
    Create a unit vector

    :param v: n-dimensional vector as a list, dict, or a numpy array, row or column vector
    :return: a unit-vector parallel to V.
    :rtype: numpy.ndarray
    :raises ValueError: for zero length vector
    
    ``unit(v)`` is a vector parallel to `v` of unit length.
    
    :seealso: norm
    
    """
    
    v = argcheck.getvector(v)
    n = np.linalg.norm(v)
    
    if n > 100*np.finfo(np.float64).eps: # if greater than eps
        return v / n
    else:
        raise ValueError("Vector has zero norm")

def norm(v):
    """
    Norm of vector

    :param v: n-vector as a list, dict, or a numpy array, row or column vector
    :return: norm of vector
    :rtype: float
    
    ``norm(v)`` is the 2-norm (length or magnitude) of the vector ``v``.
    
    :seealso: unit
    
    """
    return np.linalg.norm(v)

def isunitvec(v, tol=10):
    """
    Test if vector has unit length
    
    :param v: vector to test
    :type v: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool
        
    :seealso: unit, isunittwist
    """
    return abs(np.linalg.norm(v)-1) < tol*np.finfo(np.float64).eps

def iszerovec(v, tol=10):
    """
    Test if vector has zero length
    
    :param v: vector to test
    :type v: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has zero length
    :rtype: bool
        
    :seealso: unit, isunittwist
    """
    return np.linalg.norm(v) < tol*np.finfo(np.float64).eps


def isunittwist(v, tol=10):
    r"""
    Test if vector represents a unit twist in SE(2) or SE(3)
    
    :param v: vector to test
    :type v: list, tuple, or numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether vector has unit length
    :rtype: bool
    
    Vector is is intepretted as :math:`[v, \omega]` where :math:`v \in \mathbb{R}^n` and
    :math:`\omega \in \mathbb{R}^1` for SE(2) and :math:`\omega \in \mathbb{R}^3` for SE(3).
    
    A unit twist can be a:
        
    - unit rotational twist where :math:`|| \omega || = 1`, or
    - unit translational twist where :math:`|| \omega || = 0` and :math:`|| v || = 1`.
        
    :seealso: unit, isunitvec
    """
    v = argcheck.getvector(v)
    
    if len(v) == 3:
        # test for SE(2) twist
        return isunitvec(v[2], tol=tol) or (np.abs(v[2]) < tol*np.finfo(np.float64).eps and isunitvec(v[0:2], tol=tol))
    elif len(v) == 6:
        # test for SE(3) twist
        return isunitvec(v[3:6], tol=tol) or (np.linalg.norm(v[3:6]) < tol*np.finfo(np.float64).eps and isunitvec(v[0:3], tol=tol))
    else:
        raise ValueError

# ---------------------------------------------------------------------------------------#
def r2t(R, check=False):
    """
    Convert SO(n) to SE(n)

    :param R: rotation matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)

    ``T = r2t(R)`` is an SE(2) or SE(3) homogeneous transform equivalent to an
    SO(2) or SO(3) orthonormal rotation matrix ``R`` with a zero translational
    component
    
    - if ``R`` is 2x2 then ``T`` is 3x3: SO(2) -> SE(2)
    - if ``R`` is 3x3 then ``T`` is 4x4: SO(3) -> SE(3)
    
    :seealso: t2r, rt2tr
    """
    
    assert isinstance(R, np.ndarray)
    dim = R.shape
    assert dim[0] == dim[1], 'Matrix must be square'
    
    if check and np.abs(np.linalg.det(R) - 1) < 100*np.finfo(np.float64).eps:
        raise ValueError('Invalid rotation matrix ')
    
    T  = np.pad( R, (0,1) )
    T[-1,-1] = 1.0
        
    return T


# ---------------------------------------------------------------------------------------#
def t2r(T, check=False):
    """
    Convert SE(n) to SO(n)

    :param T: homogeneous transformation matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: rotation matrix
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)


    ``R = T2R(T)`` is the orthonormal rotation matrix component of homogeneous
    transformation matrix ``T``
    
    - if ``T`` is 3x3 then ``R`` is 2x2: SE(2) -> SO(2)
    - if ``T`` is 4x4 then ``R`` is 3x3: SE(3) -> SO(3)
    
    Any translational component of T is lost.
    
    :seealso: r2t, tr2rt
    """
    assert isinstance(T, np.ndarray)
    dim = T.shape
    assert dim[0] == dim[1], 'Matrix must be square'
    
    if dim[0] == 3:
        R = T[:2,:2]
    elif dim[0] == 4:
        R = T[:3,:3]
    else:
        raise ValueError('Value must be a rotation matrix')
    
    if check and isR(R):
            raise ValueError('Invalid rotation matrix')

    return R

# ---------------------------------------------------------------------------------------#
def tr2rt(T, check=False):
    """
    Convert SE(3) to SO(3) and translation

    :param T: homogeneous transform matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: Rotation matrix and translation vector
    :rtype: tuple: numpy.ndarray, shape=(2,2) or (3,3); numpy.ndarray, shape=(2,) or (3,)

    (R,t) = tr2rt(T) splits a homogeneous transformation matrix (NxN) into an orthonormal
    rotation matrix R (MxM) and a translation vector T (Mx1), where N=M+1.
        
    - if ``T`` is 3x3 - in SE(2) - then ``R`` is 2x2 and ``t`` is 2x1.
    - if ``T`` is 4x4 - in SE(3) - then ``R`` is 3x3 and ``t`` is 3x1.
    
    :seealso: rt2tr, tr2r
    """
    dim = T.shape
    assert dim[0] == dim[1], 'Matrix must be square'
    
    if dim[0] == 3:
        R = t2r(T, check)
        t = T[:2,2]
    elif dim[0] == 4:
        R = t2r(T, check)
        t = T[:3,3]
    else:
        raise ValueError('T must be an SE2 or SE3 homogeneous transformation matrix')
    
    return [R, t]

# ---------------------------------------------------------------------------------------#
def rt2tr(R, t, check=False):
    """
    Convert SO(3) and translation to SE(3)

    :param R: rotation matrix
    :param t: translation vector
    :param check: check if rotation matrix is valid (default False, no check)
    :return: homogeneous transform
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)

    ``T = rt2tr(R, t)`` is a homogeneous transformation matrix (N+1xN+1) formed from an
    orthonormal rotation matrix ``R`` (NxN) and a translation vector ``t``
    (Nx1).
        
    - If ``R`` is 2x2 and ``t`` is 2x1, then ``T`` is 3x3
    - If ``R`` is 3x3 and ``t`` is 3x1, then ``T`` is 4x4
    
    :seealso: tr2rt, r2t
    """
    t = argcheck.getvector(t, dim=None, out='array')
    if R.shape[0] != t.shape[0]:
        raise ValueError("R and t must have the same number of rows")
    if check and np.abs(np.linalg.det(R) - 1) < 100*np.finfo(np.float64).eps:
            raise ValueError('Invalid rotation matrix')
            
    if R.shape == (2, 2):
        T = np.eye(3)
        T[:2,:2] = R
        T[:2,2] = t
    elif R.shape == (3, 3):
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
    else:
        raise ValueError('R must be an SO2 or SO3 rotation matrix')
    
    return T

#======================= predicates

def isR(R, tol = 10):
    r"""
    Test if matrix belongs to SO(n)
    
    :param R: matrix to test
    :type R: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper orthonormal rotation matrix
    :rtype: bool
    
    Checks orthogonality, ie. :math:`{\bf R} {\bf R}^T = {\bf I}` and :math:`\det({\bf R}) > 0`.
    For the first test we check that the norm of the residual is less than ``tol * eps``.
    
    :seealso: isrot2, isrot
    """
    return np.linalg.norm( R@R.T - np.eye(R.shape[0]) ) < tol*np.finfo(np.float64).eps \
        and np.linalg.det( R@R.T ) > 0

def ishom(T, check=False):
    """
    Test if matrix belongs to SE(3)
    
    :param T: matrix to test
    :type T: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SE(3) homogeneous transformation matrix
    :rtype: bool
    
    - ``ISHOM(T)`` is True if the argument ``T`` is of dimension 4x4
    - ``ISHOM(T, check=True)`` as above, but also checks orthogonality of the rotation sub-matrix and 
      validitity of the bottom row.
    
    :seealso: isR, isrot, ishom2
    """
    return T.shape == (4,4) and (not check or (isR(T[:3,:3]) and np.all(T[3,:] == np.array([0,0,0,1]))))

def ishom2(T, check=False):
    """
    Test if matrix belongs to SE(2)
    
    :param T: matrix to test
    :type T: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SE(2) homogeneous transformation matrix
    :rtype: bool
    
    - ``ISHOM2(T)`` is True if the argument ``T`` is of dimension 3x3
    - ``ISHOM2(T, check=True)`` as above, but also checks orthogonality of the rotation sub-matrix and 
      validitity of the bottom row.
    
    :seealso: isR, isrot2, ishom, isvec
    """
    return T.shape == (3,3) and (not check or (isR(T[:2,:2]) and np.all(T[2,:] == np.array([0,0,1]))))

def isrot(R, check=False):
    """
    Test if matrix belongs to SO(3)
    
    :param R: matrix to test
    :type R: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SO(3) rotation matrix
    :rtype: bool
    
    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 3x3
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.
    
    :seealso: isR, isrot2, ishom
    """
    return R.shape == (3,3) and (not check or isR(R))

def isrot2(R, check=False):
    """
    Test if matrix belongs to SO(2)
    
    :param R: matrix to test
    :type R: numpy.ndarray
    :param check: check validity of rotation submatrix
    :type check: bool
    :return: whether matrix is an SO(2) rotation matrix
    :rtype: bool
    
    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 2x2
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.
    
    :seealso: isR, ishom2, isrot
    """
    return R.shape == (2,2) and (not check or isR(R))

def isskew(S, tol = 10):
    r"""
    Test if matrix belongs to so(n)
    
    :param S: matrix to test
    :type S: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool
    
    Checks skew-symmetry, ie. :math:`{\bf S} + {\bf S}^T = {\bf 0}`.
    We check that the norm of the residual is less than ``tol * eps``.
    
    :seealso: isskewa
    """
    return np.linalg.norm(S + S.T) < tol*np.finfo(np.float64).eps

def isskewa(S, tol = 10):
    r"""
    Test if matrix belongs to se(n)
    
    :param S: matrix to test
    :type S: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool
    
    Check if matrix is augmented skew-symmetric, ie. the top left (n-1xn-1) partition ``S`` is
    skew-symmetric :math:`{\bf S} + {\bf S}^T = {\bf 0}`, and the bottom row is zero
    We check that the norm of the residual is less than ``tol * eps``.
    
    :seealso: isskew
    """
    return np.linalg.norm(S[0:-1,0:-1] + S[0:-1,0:-1].T) < tol*np.finfo(np.float64).eps \
        and np.all(S[-1,:] == 0)

def iseye(S, tol = 10):
    """
    Test if matrix is identity
    
    :param S: matrix to test
    :type S: numpy.ndarray
    :param tol: tolerance in units of eps
    :type tol: float
    :return: whether matrix is a proper skew-symmetric matrix
    :rtype: bool
    
    Check if matrix is an identity matrix. We test that the trace tom row is zero
    We check that the norm of the residual is less than ``tol * eps``.
    
    :seealso: isskew, isskewa
    """
    s = S.shape
    if len(s) != 2 or s[0] != s[1]:
        return False  # not a square matrix
    return abs(S.trace() - s[0]) < tol * np.finfo(np.float64).eps
        
    
    
#========================= angle sequences

# ---------------------------------------------------------------------------------------#
def rpy2r(roll, pitch=None, yaw=None, unit='rad', order='zyx'):
    """
    Create an SO(3) rotation matrix from roll-pitch-yaw angles

    :param roll: roll angle
    :type roll: float
    :param pitch: pitch angle
    :type pitch: float
    :param yaw: yaw angle
    :type yaw: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)

    - ``rpy2r(ROLL, PITCH, YAW)`` is an SO(3) orthonormal rotation matrix
      (3x3) equivalent to the specified roll, pitch, yaw angles angles.
      These correspond to successive rotations about the axes specified by ``order``:
          
        - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
          then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
          and y-axis sideways.
        - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
          then by roll about the new z-axis. Covention for a robot gripper with z-axis forward
          and y-axis between the gripper fingers.
        - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
          then by roll about the new z-axis. Convention for a camera with z-axis parallel
          to the optic axis and x-axis parallel to the pixel rows.
          
    - ``rpy2r(RPY)`` as above but the roll, pitch, yaw angles are taken
      from ``RPY`` which is a 3-vector (list, tuple, numpy.ndarray) with values
      (ROLL, PITCH, YAW). 
    """
    
    if np.isscalar(roll):
        angles = [roll, pitch, yaw]
    else:
        angles = argcheck.getvector(roll, 3)
        
    angles = argcheck.getunit(angles, unit)
    
    if order == 'xyz' or order == 'arm':
        R = rotx(angles[2]) @ roty(angles[1]) @ rotz(angles[0])
    elif order == 'zyx' or order == 'vehicle':
        R = rotz(angles[2]) @ roty(angles[1]) @ rotx(angles[0])
    elif order == 'yxz' or order == 'camera':
        R = roty(angles[2]) @ rotx(angles[1]) @ rotz(angles[0])
    else:
        raise ValueError('Invalid angle order')
        
    return R


# ---------------------------------------------------------------------------------------#
def rpy2tr(roll, pitch=None, yaw=None, unit='rad', order='zyx'):
    """
    Create an SE(3) rotation matrix from roll-pitch-yaw angles

    :param roll: roll angle
    :type roll: float
    :param pitch: pitch angle
    :type pitch: float
    :param yaw: yaw angle
    :type yaw: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param unit: rotation order: 'zyx' [default], 'xyz', or 'yxz'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)

    - ``rpy2tr(ROLL, PITCH, YAW)`` is an SO(3) orthonormal rotation matrix
      (3x3) equivalent to the specified roll, pitch, yaw angles angles.
      These correspond to successive rotations about the axes specified by ``order``:
          
        - 'zyx' [default], rotate by yaw about the z-axis, then by pitch about the new y-axis,
          then by roll about the new x-axis.  Convention for a mobile robot with x-axis forward
          and y-axis sideways.
        - 'xyz', rotate by yaw about the x-axis, then by pitch about the new y-axis,
          then by roll about the new z-axis. Covention for a robot gripper with z-axis forward
          and y-axis between the gripper fingers.
        - 'yxz', rotate by yaw about the y-axis, then by pitch about the new x-axis,
          then by roll about the new z-axis. Convention for a camera with z-axis parallel
          to the optic axis and x-axis parallel to the pixel rows.
          
    - ``rpy2tr(RPY)`` as above but the roll, pitch, yaw angles are taken
      from ``RPY`` which is a 3-vector (list, tuple, numpy.ndarray) with values
      (ROLL, PITCH, YAW). 
    
    Notes:
        
    - The translational part is zero.
    """

    R = rpy2r(roll, pitch, yaw, order=order, unit=unit)
    return r2t(R)

# ---------------------------------------------------------------------------------------#
def eul2r(phi, theta=None, psi=None, unit='rad'):
    """
    Create an SO(3) rotation matrix from Euler angles

    :param phi: Z-axis rotation
    :type phi: float
    :param theta: Y-axis rotation
    :type theta: float
    :param psi: Z-axis rotation
    :type psi: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)

    - ``R = eul2r(PHI, THETA, PSI)`` is an SO(3) orthonornal rotation
      matrix equivalent to the specified Euler angles.  These correspond
      to rotations about the Z, Y, Z axes respectively.
    - ``R = eul2r(EUL)`` as above but the Euler angles are taken from
      ``EUL`` which is a 3-vector (list, tuple, numpy.ndarray) with values
      (PHI THETA PSI). 
    """
    
    if np.isscalar(phi):
        angles = [phi, theta, psi]
    else:
        angles = argcheck.getvector(phi, 3)
        
    angles = argcheck.getunit(angles, unit)
        
    return rotz(angles[0]) @ roty(angles[1]) @ rotz(angles[2])


# ---------------------------------------------------------------------------------------#
def eul2tr(phi, theta=None, psi=None, unit='rad'):
    """
    Create an SE(3) pure rotation matrix from Euler angles

    :param phi: Z-axis rotation
    :type phi: float
    :param theta: Y-axis rotation
    :type theta: float
    :param psi: Z-axis rotation
    :type psi: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpdy.ndarray, shape=(4,4)

    - ``R = eul2tr(PHI, THETA, PSI)`` is an SE(3) homogeneous transformation
      matrix equivalent to the specified Euler angles.  These correspond
      to rotations about the Z, Y, Z axes respectively.
    - ``R = eul2tr(EUL)`` as above but the Euler angles are taken from 
      ``EUL`` which is a 3-vector (list, tuple, numpy.ndarray) with values
      (PHI THETA PSI). 
    
    Notes:
        
    - The translational part is zero.
    """
    
    R = eul2r(phi, theta, psi, unit=unit)
    return r2t(R)

# ---------------------------------------------------------------------------------------#
def angvec2r(theta, v, unit='rad'):
    """
    Create an SO(3) rotation matrix from rotation angle and axis

    :param theta: rotation
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param v: rotation axis
    :type v: 3-vector: list, tuple, numpy.ndarray
    :return: 3x3 rotation matrix
    :rtype: numpdy.ndarray, shape=(3,3)
    
    ``angvec2r(THETA, V)`` is an SO(3) orthonormal rotation matrix
    equivalent to a rotation of ``THETA`` about the vector ``V``.
    
    Notes:
        
    - If ``THETA == 0`` then return identity matrix.
    - If ``THETA ~= 0`` then ``V`` must have a finite length.
    """
    assert np.isscalar(theta) and argcheck.isvector(v, 3), "Arguments must be theta and vector"
    
    if np.linalg.norm(v) < 10*np.finfo(np.float64).eps:
            return np.eye(3)
        
    theta = argcheck.getunit(theta, unit)
    
    # Rodrigue's equation

    sk = transforms.skew( transforms.unit(v) )
    R = np.eye(3) + math.sin(theta) * sk + (1.0 - math.cos(theta)) * sk @ sk
    return R


# ---------------------------------------------------------------------------------------#
def angvec2tr(theta, v, unit='rad'):
    """
    Create an SE(3) pure rotation from rotation angle and axis

    :param theta: rotation
    :type theta: float
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :param v: rotation axis
    :type v: 3-vector: list, tuple, numpy.ndarray
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpdy.ndarray, shape=(4,4)
    
    ``angvec2tr(THETA, V)`` is an SE(3) homogeneous transformation matrix
    equivalent to a rotation of ``THETA`` about the vector ``V``.
    
    Notes:
        
    - If ``THETA == 0`` then return identity matrix.
    - If ``THETA ~= 0`` then ``V`` must have a finite length.
    - The translational part is zero.
    """
    return r2t(angvec2r(theta, v, unit=unit))


# ---------------------------------------------------------------------------------------#
def oa2r(o, a=None):
    """
    Create SO(3) rotation matrix from two vectors

    :param o: 3-vector parallel to Y- axis
    :type o: list, tuple, numpy.ndarray
    :param a: 3-vector parallel to the Z-axis
    :type o: list, tuple, numpy.ndarray
    :return: 3x3 rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    ``T = oa2tr(O, A)`` is an SO(3) orthonormal rotation matrix for a frame defined in terms of
    vectors parallel to its Y- and Z-axes with respect to a reference frame.  In robotics these axes are 
    respectively called the orientation and approach vectors defined such that
    R = [N O A] and N = O x A.
    
    Steps:
        
        1. N' = O x A
        2. O' = A x N
        3. normalize N', O', A
        4. stack horizontally into rotation matrix

    Notes:
        
    - The A vector is the only guaranteed to have the same direction in the resulting 
      rotation matrix
    - O and A do not have to be unit-length, they are normalized
    - O and A do not have to be orthogonal, so long as they are not parallel
    - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.
    """
    o = argcheck.getvector(o, 3, out='array')
    a = argcheck.getvector(a, 3, out='array')
    n = np.cross(o, a)
    o = np.cross(a, n)
    R = np.stack( (transforms.unit(n), transforms.unit(o), transforms.unit(a)), axis=1)
    return R


# ---------------------------------------------------------------------------------------#
def oa2tr(o, a=None):
    """
    Create SE(3) pure rotation from two vectors

    :param o: 3-vector parallel to Y- axis
    :type o: list, tuple, numpy.ndarray
    :param a: 3-vector parallel to the Z-axis
    :type o: list, tuple, numpy.ndarray
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    ``T = oa2tr(O, A)`` is an SE(3) homogeneous transformation matrix for a frame defined in terms of
    vectors parallel to its Y- and Z-axes with respect to a reference frame.  In robotics these axes are 
    respectively called the orientation and approach vectors defined such that
    R = [N O A] and N = O x A.
    
    Steps:
        
        1. N' = O x A
        2. O' = A x N
        3. normalize N', O', A
        4. stack horizontally into rotation matrix

    Notes:
        
    - The A vector is the only guaranteed to have the same direction in the resulting 
      rotation matrix
    - O and A do not have to be unit-length, they are normalized
    - O and A do not have to be orthogonal, so long as they are not parallel
    - The translational part is zero.
    - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.
    """
    return r2t(oa2r(o, a))


# ------------------------------------------------------------------------------------------------------------------- #
def tr2angvec(T, unit='rad', check=False):
    r"""
    Convert SO(3) or SE(3) to angle and rotation vector
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param check: check that rotation matrix is valid
    :type check: bool
    :return: :math:`(\theta, {\bf v})`
    :rtype: float, numpy.ndarray, shape=(3,)
    
    ``tr2angvec(R)`` is a rotation angle and a vector about which the rotation
    acts that corresponds to the rotation part of ``R``. 
        
    By default the angle is in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: angvec2r, angvec2tr, tr2rpy, tr2eul
    
    """

    if argcheck.ismatrix(T, (4,4)):
        R = t2r(T)
    else:
        R = T
    assert isrot(R, check=check)
    
    v = vex( trlog(R) )
    
    theta = np.linalg.norm(v)
    v = v / theta
    
    if unit == 'deg':
        theta *= 180 / math.pi
               
    return (theta, v)


# ------------------------------------------------------------------------------------------------------------------- #
def tr2eul(T, unit='rad', flip=False, check=False):
    r"""
    Convert SO(3) or SE(3) to ZYX Euler angles
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param flip: choose first Euler angle to be in quadrant 2 or 3
    :type flip: bool
    :param check: check that rotation matrix is valid
    :type check: bool
    :return: ZYZ Euler angles
    :rtype: numpy.ndarray, shape=(3,)
    
    ``tr2eul(R)`` are the Euler angles corresponding to 
    the rotation part of ``R``. 
    
    The 3 angles :math:`[\phi, \theta, \psi` correspond to sequential rotations about the
    Z, Y and Z axes respectively.
    
    By default the angles are in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - There is a singularity for the case where :math:`\theta=0` in which case :math:`\phi` is arbitrarily set to zero and :math:`\phi` is set to :math:`\phi+\psi`.
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: eul2r, eul2tr, tr2rpy, tr2angvec
    """
    
    if argcheck.ismatrix(T, (4,4)):
        R = t2r(T)
    else:
        R = T
    assert isrot(R, check=check)
    
    eul = np.zeros((3,))
    if abs(R[0,2]) < 10*np.finfo(np.float64).eps and abs(R[1,2]) < 10*np.finfo(np.float64).eps:
        eul[0] = 0
        sp = 0
        cp = 1
        eul[1] = math.atan2(cp * R[0,2] + sp * R[1,2],R[2,2])
        eul[2] = math.atan2(-sp * R[0,0] + cp * R[1,0], -sp * R[0,1] + cp * R[1,1])
    else:
        if flip:
            eul[0] = math.atan2(-R[1,2], -R[0,2])
        else:
            eul[0] = math.atan2(R[1,2],R[0,2])
        sp = math.sin(eul[0])
        cp = math.cos(eul[0])
        eul[1] = math.atan2(cp * R[0,2] + sp * R[1,2], R[2,2])
        eul[2] = math.atan2(-sp * R[0,0] + cp * R[1,0], -sp * R[0,1] + cp * R[1, 1])

    if unit == 'deg':
        eul *= 180 / math.pi

    return eul

# ------------------------------------------------------------------------------------------------------------------- #
def tr2rpy(T, unit='rad', order='zyx', check=False):
    """
    Convert SO(3) or SE(3) to roll-pitch-yaw angles
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param order: 'xyz', 'zyx' or 'yxz' [default 'zyx']
    :type unit: str
    :param check: check that rotation matrix is valid
    :type check: bool
    :return: Roll-pitch-yaw angles
    :rtype: numpy.ndarray, shape=(3,)
    
    ``tr2rpy(R)`` are the roll-pitch-yaw angles corresponding to 
    the rotation part of ``R``. 
    
    The 3 angles RPY=[R,P,Y] correspond to sequential rotations about the
    Z, Y and X axes respectively.  The axis order sequence can be changed by
    setting:
 
    - `order='xyz'`  for sequential rotations about X, Y, Z axes
    - `order='yxz'`  for sequential rotations about Y, X, Z axes
    
    By default the angles are in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - There is a singularity for the case where P=:math:`\pi/2` in which case R is arbitrarily set to zero and Y is the sum (R+Y).
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: rpy2r, rpy2tr, tr2eul, tr2angvec
    """
    
    if argcheck.ismatrix(T, (4,4)):
        R = t2r(T)
    else:
        R = T
    assert isrot(R, check=check)
    
    rpy = np.zeros((3,))
    if order == 'xyz' or order == 'arm':

        # XYZ order
        if abs(abs(R[0,2]) - 1) < 10*np.finfo(np.float64).eps:  # when |R13| == 1
            # singularity
            rpy[0] = 0  # roll is zero
            if R[0,2] > 0:
                rpy[2] = math.atan2( R[2,1], R[1,1])   # R+Y
            else:
                rpy[2] = -math.atan2( R[1,0], R[2,0])   # R-Y
            rpy[1] = math.asin(R[0,2])
        else:
            rpy[0] = -math.atan2(R[0,1], R[0,0])
            rpy[2] = -math.atan2(R[1,2], R[2,2])
            
            k = np.argmax(np.abs( [R[0,0], R[0,1], R[1,2], R[2,2]] ))
            if k == 0:
                rpy[1] =  math.atan(R[0,2]*math.cos(rpy[0])/R[0,0])
            elif k == 1:
                rpy[1] = -math.atan(R[0,2]*math.sin(rpy[0])/R[0,1])
            elif k == 2:
                rpy[1] = -math.atan(R[0,2]*math.sin(rpy[2])/R[1,2])
            elif k == 3:
                rpy[1] =  math.atan(R[0,2]*math.cos(rpy[2])/R[2,2])
                                        

    elif order == 'zyx' or order == 'vehicle':

        # old ZYX order (as per Paul book)
        if abs(abs(R[2,0]) - 1) < 10*np.finfo(np.float64).eps:  # when |R31| == 1
            # singularity
            rpy[0] = 0     # roll is zero
            if R[2,0] < 0:
                rpy[2] = -math.atan2(R[0,1], R[0,2])  # R-Y
            else:
                rpy[2] = math.atan2(-R[0,1], -R[0,2])  # R+Y
            rpy[1] = -math.asin(R[2,0])
        else:
            rpy[0] = math.atan2(R[2,1], R[2,2])  # R
            rpy[2] = math.atan2(R[1,0], R[0,0])  # Y
                 
            k = np.argmax(np.abs( [R[0,0], R[1,0], R[2,1], R[2,2]] ))
            if k == 0:
                rpy[1] = -math.atan(R[2,0]*math.cos(rpy[2])/R[0,0])
            elif k == 1:
                rpy[1] = -math.atan(R[2,0]*math.sin(rpy[2])/R[1,0])
            elif k == 2:
                rpy[1] = -math.atan(R[2,0]*math.sin(rpy[0])/R[2,1])
            elif k == 3:
                rpy[1] = -math.atan(R[2,0]*math.cos(rpy[0])/R[2,2])

    elif order == 'yxz' or order == 'camera':

            if abs(abs(R[1,2]) - 1) < 10*np.finfo(np.float64).eps:  # when |R23| == 1
                # singularity
                rpy[0] = 0
                if R[1,2] < 0:
                    rpy[2] = -math.atan2(R[2,0], R[0,0])   # R-Y
                else:
                    rpy[2] = math.atan2(-R[2,0], -R[2,1])   # R+Y
                rpy[1] = -math.asin(R[1,2])    # P
            else:
                rpy[0] = math.atan2(R[1,0], R[1,1])
                rpy[2] = math.atan2(R[0,2], R[2,2])
                
                k = np.argmax(np.abs( [R[1,0], R[1,1], R[0,2], R[2,2]] ))
                if k == 0:
                    rpy[1] = -math.atan(R[1,2]*math.sin(rpy[0])/R[1,0])
                elif k == 1:
                    rpy[1] = -math.atan(R[1,2]*math.cos(rpy[0])/R[1,1])
                elif k == 2:
                    rpy[1] = -math.atan(R[1,2]*math.sin(rpy[2])/R[0,2])
                elif k == 3:
                    rpy[1] = -math.atan(R[1,2]*math.cos(rpy[2])/R[2,2])  

    else:
        raise ValueError('Invalid order')

    if unit == 'deg':
        rpy *= 180 / math.pi

    return rpy


# ---------------------------------------------------------------------------------------#
def skew(v):
    r"""
    Create skew-symmetric metrix from vector

    :param v: 1- or 3-vector
    :type v: list, tuple or numpy.ndarray
    :return: skew-symmetric matrix in so(2) or so(3)
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)
    :raises: ValueError
    
    ``skew(V)`` is a skew-symmetric matrix formed from the elements of ``V``.

    - ``len(V)``  is 1 then ``S`` = :math:`\left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]`
    - ``len(V)`` is 3 then ``S`` = :math:`\left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]`

    Notes:
        
    - This is the inverse of the function ``vex()``.
    - These are the generator matrices for the Lie algebras so(2) and so(3).
    
    :seealso: vex, skewa
    """
    v = argcheck.getvector(v, None, 'sequence')
    if len(v) == 1:
        s = np.array([
                [0, -v[0]], 
                [v[0], 0]   ])
    elif len(v) == 3:
        s = np.array([
                [ 0,   -v[2], v[1]], 
                [ v[2], 0,   -v[0]], 
                [-v[1], v[0], 0]    ])
    else:
        raise AttributeError("argument must be a 1- or 3-vector")
        
    return s

# ---------------------------------------------------------------------------------------#
def vex(s):
    r"""
    Convert skew-symmetric matrix to vector

    :param s: skew-symmetric matrix
    :type s: numpy.ndarray, shape=(2,2) or (3,3)
    :return: vector of unique values
    :rtype: numpy.ndarray, shape=(1,) or (3,)
    :raises: ValueError

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.
    
    - ``S`` is 2x2 - so(2) case - where ``S`` :math:`= \left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]` then return :math:`[v]`
    - ``S`` is 3x3 - so(3) case -  where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]` then return :math:`[v_x, v_y, v_z]`.
    
    Notes:
        
    - This is the inverse of the function ``skew()``.
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
      is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
      element of the matrix.
      
    :seealso: skew, vexa
    """
    if s.shape == (3,3):
        return 0.5 * np.array([s[2, 1] - s[1, 2], s[0, 2] - s[2, 0], s[1, 0] - s[0, 1]])
    elif s.shape == (2, 2):
        return 0.5 * np.array([s[1, 0] - s[0, 1]])
    else:
        raise ValueError("Argument must be 2x2 or 3x3 matrix")

# ---------------------------------------------------------------------------------------#
def skewa(v):
    r"""
    Create augmented skew-symmetric metrix from vector

    :param v: 3- or 6-vector
    :type v: list, tuple or numpy.ndarray
    :return: augmented skew-symmetric matrix in se(2) or se(3)
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)
    :raises: ValueError
    
    ``skewa(V)`` is an augmented skew-symmetric matrix formed from the elements of ``V``.

    - ``len(V)`` is 3 then S = :math:`\left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]`
    - ``len(V)`` is 6 then S = :math:`\left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]`

    Notes:
        
    - This is the inverse of the function ``vexa()``.
    - These are the generator matrices for the Lie algebras se(2) and se(3).
    - Map twist vectors in 2D and 3D space to se(2) and se(3).
    
    :seealso: vexa, skew
    """
    
    v = argcheck.getvector(v, None, 'sequence')
    if len(v) == 3:
        omega = np.zeros((3,3))
        omega[:2,:2] = skew(v[2])
        omega[:2,2] = v[0:2]
        return omega
    elif len(v) == 6:
        omega = np.zeros((4,4))
        omega[:3,:3] = skew(v[3:6])
        omega[:3,3] = v[0:3]
        return omega
    else:
        raise AttributeError("expecting a 3- or 6-vector")

def vexa(Omega):
    r"""
    Convert skew-symmetric matrix to vector

    :param s: augmented skew-symmetric matrix
    :type s: numpy.ndarray, shape=(3,3) or (4,4)
    :return: vector of unique values
    :rtype: numpy.ndarray, shape=(3,) or (6,)
    :raises: ValueError

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.
    
    - ``S`` is 3x3 - se(2) case - where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]` then return :math:`[v_1, v_2, v_3]`.
    - ``S`` is 4x4 - se(3) case -  where ``S`` :math:`= \left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]` then return :math:`[v_1, v_2, v_3, v_4, v_5, v_6]`.
    
    
    Notes:
        
    - This is the inverse of the function ``skewa``.
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
      is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
      element of the matrix.
      
    :seealso: skewa, vex
    """
    if Omega.shape == (4,4):
        return np.hstack( (transl(Omega), vex(t2r(Omega))) )
    elif Omega.shape == (3,3):
        return np.hstack( (transl2(Omega), vex(t2r(Omega))) )
    else:
        raise AttributeError("expecting a 3x3 or 4x4 matrix")
        


# ---------------------------------------------------------------------------------------#
def trlog(T, check=True):
    """
    Logarithm of SO(3) or SE(3) matrix

    :param T: SO(3) or SE(3) matrix
    :type T: numpy.ndarray, shape=(3,3) or (4,4)
    :return: logarithm
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)
    :raises: ValueError

    An efficient closed-form solution of the matrix logarithm for arguments that are SO(3) or SE(3).
    
    - ``trlog(R)`` is the logarithm of the passed rotation matrix ``R`` which will be 
      3x3 skew-symmetric matrix.  The equivalent vector from ``vex()`` is parallel to rotation axis
      and its norm is the amount of rotation about that axis.
    - ``trlog(T)`` is the logarithm of the passed homogeneous transformation matrix ``T`` which will be 
      4x4 augumented skew-symmetric matrix. The equivalent vector from ``vexa()`` is the twist
      vector (6x1) comprising [v w].


    :seealso: trexp, vex, vexa
    """
    
    if ishom(T, check=check):
        # SE(3) matrix


        if iseye(T):
            # is identity matrix
            return np.zeros((4,4))
        else:
            [R,t] = tr2rt(T)
            S = trlog(R, check=False)  # recurse
            w = vex(S)
            theta = norm(w)
            skw = S / theta
            Ginv = np.eye(3) / theta - skw / 2 + (1 / theta - 1 / np.tan(theta / 2) / 2) * skw ** 2
            v = Ginv * t
            return rt2tr(skw, v)
        
    elif isrot(T, check=check):
        # deal with rotation matrix
        R = T
        if iseye(R):
            # matrix is identity
            return np.zeros((3,3))
        elif abs(R[0,0] + 1) < 100 * np.finfo(np.float64).eps:
            # tr R = -1
            # rotation by +/- pi, +/- 3pi etc.
            mx = R.diagonal().max()
            k = R.diagonal().argmax()
            I = np.eye(3)
            col = R[:, k] + I[:, k]
            w = col / np.sqrt(2 * (1 + mx))
            theta = np.pi
            return skew(w*theta)
        else:
            # general case
            theta = np.arccos((R[0,0] - 1) / 2)
            skw = (R - R.T) / 2 / np.sin(theta)
            return skw * theta
    else:
        raise ValueError("Expect SO(3) or SE(3) matrix")

# ---------------------------------------------------------------------------------------#
def trexp(S, theta=None):
    """
    Exponential of so(3) or se(3) matrix

    :param S: so(3), se(3) matrix or equivalent velctor
    :type T: numpy.ndarray, shape=(3,3), (3,), (4,4), or (6,)
    :param theta: motion
    :type theta: float
    :return: 3x3 or 4x4 matrix exponential in SO(3) or SE(3)
    :rtype: numpy.ndarray, shape=(3,3) or (4,4)
    
    An efficient closed-form solution of the matrix exponential for arguments
    that are so(3) or se(3).
    
    For so(3) the results is an SO(3) rotation matrix:

    - ``trexp(S)`` is the matrix exponential of the so(3) element ``S`` which is a 3x3
       skew-symmetric matrix.
    - ``trexp(S, THETA)`` as above but for an so(3) motion of S*THETA, where ``S`` is
      unit-norm skew-symmetric matrix representing a rotation axis and a rotation magnitude
      given by ``THETA``.
    - ``trexp(W)`` is the matrix exponential of the so(3) element ``W`` expressed as
      a 3-vector (list, tuple, numpy.ndarray).
    - ``trexp(W, THETA)`` as above but for an so(3) motion of W*THETA where ``W`` is a
      unit-norm vector representing a rotation axis and a rotation magnitude
      given by ``THETA``. ``W`` is expressed as a 3-vector (list, tuple, numpy.ndarray).


    For se(3) the results is an SE(3) homogeneous transformation matrix:

    - ``trexp(SIGMA)`` is the matrix exponential of the se(3) element ``SIGMA`` which is
      a 4x4 augmented skew-symmetric matrix.
    - ``trexp(SIGMA, THETA)`` as above but for an se(3) motion of SIGMA*THETA, where ``SIGMA``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
    - ``trexp(TW)`` is the matrix exponential of the se(3) element ``TW`` represented as
      a 6-vector which can be considered a screw motion.
    - ``trexp(TW, THETA)`` as above but for an se(3) motion of TW*THETA, where ``TW``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
     
     :seealso: trlog, trexp2
    """
   
    if argcheck.ismatrix(S, (4,4)) or argcheck.isvector(S, 6):
        # se(3) case
        if argcheck.ismatrix(S, (4,4)):
            # augmentented skew matrix
            tw = vexa(S)
        else:
            # 6 vector
            tw = argcheck.getvector(S)

        if theta is not None:
                assert isunittwist(tw), 'If theta is specified S must be a unit twist'
                
        t = tw[0:3]
        w = tw[3:6]
        
    elif argcheck.ismatrix(S, (3,3)) or argcheck.isvector(S, 3):
        # so(3) case
        if argcheck.ismatrix(S, (3,3)):
            # skew symmetric matrix
            w = vex(S)
        else:
            # 3 vector
            w = argcheck.getvector(S)
            
        if theta is not None:
            assert isunitvec(w), 'If theta is specified S must be a unit twist'
        t = None
    else:
        raise ValueError(" First argument must be SO(3), 3-vector, SE(3) or 6-vector")
    
    
    # do Rodrigues' formula for rotation
    if iszerovec(w):
        # for a zero so(3) return unit matrix, theta not relevant
        R = np.eye(3)
        V = np.eye(3)
    else:
        if theta is None:
            #  theta is not given, extract it
            theta = norm(w)
            w = unit(w)

        skw = skew(w)
        R = np.eye(3) + math.sin(theta) * skw + (1.0 - math.cos(theta)) * skw @ skw
        V = None
    
    if t is None:
        # so(3) case
        return R
    else:
        if V is None:
            V = np.eye(3) + (1.0-math.cos(theta))*skw/theta + (theta-math.sin(theta))/theta*skw @ skw
        return rt2tr(R, V@t)


# ---------------------------------------------------------------------------------------#
def trexp2(S, theta=None):
    """
    Exponential of so(2) or se(2) matrix

    :param S: so(2), se(2) matrix or equivalent velctor
    :type T: numpy.ndarray, shape=(2,2), (1,), (3,3), or (3,)
    :param theta: motion
    :type theta: float
    :return: 2x2 or 3x3 matrix exponential in SO(2) or SE(2)
    :rtype: numpy.ndarray, shape=(2,2) or (3,3)
    
    An efficient closed-form solution of the matrix exponential for arguments
    that are so(2) or se(2).
    
    For so(2) the results is an SO(2) rotation matrix:

    - ``trexp2(S)`` is the matrix exponential of the so(3) element ``S`` which is a 2x2
      skew-symmetric matrix.
    - ``trexp2(S, THETA)`` as above but for an so(3) motion of S*THETA, where ``S`` is
      unit-norm skew-symmetric matrix representing a rotation axis and a rotation magnitude
      given by ``THETA``.
    - ``trexp2(W)`` is the matrix exponential of the so(2) element ``W`` expressed as
      a 1-vector (list, tuple, numpy.ndarray).
    - ``trexp2(W, THETA)`` as above but for an so(3) motion of W*THETA where ``W`` is a
      unit-norm vector representing a rotation axis and a rotation magnitude
      given by ``THETA``. ``W`` is expressed as a 1-vector (list, tuple, numpy.ndarray).


    For se(2) the results is an SE(2) homogeneous transformation matrix:

    - ``trexp2(SIGMA)`` is the matrix exponential of the se(2) element ``SIGMA`` which is
      a 3x3 augmented skew-symmetric matrix.
    - ``trexp2(SIGMA, THETA)`` as above but for an se(3) motion of SIGMA*THETA, where ``SIGMA``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
    - ``trexp2(TW)`` is the matrix exponential of the se(3) element ``TW`` represented as
      a 3-vector which can be considered a screw motion.
    - ``trexp2(TW, THETA)`` as above but for an se(2) motion of TW*THETA, where ``TW``
      must represent a unit-twist, ie. the rotational component is a unit-norm skew-symmetric
      matrix.
          
     :seealso: trlog, trexp2
    """
    
    if argcheck.ismatrix(S, (3,3)) or argcheck.isvector(S, 3):
        # se(2) case
        if argcheck.ismatrix(S, (3,3)):
            # augmentented skew matrix
            tw = vexa(S)
        else:
            # 3 vector
            tw = argcheck.getvector(S)

        if theta is not None:
                assert isunittwist(tw), 'If theta is specified S must be a unit twist'
                
        t = tw[0:2]
        w = tw[2]
        
    elif argcheck.ismatrix(S, (2,2)) or argcheck.isvector(S, 1):
        # so(2) case
        if argcheck.ismatrix(S, (2,2)):
            # skew symmetric matrix
            w = vex(S)
        else:
            # 1 vector
            w = argcheck.getvector(S)
            
        if theta is not None:
            assert isunitvec(w), 'If theta is specified S must be a unit twist'
        t = None
    else:
        raise ValueError(" First argument must be SO(2), 1-vector, SE(2) or 3-vector")
    
    
    # do Rodrigues' formula for rotation
    if iszerovec(w):
        # for a zero so(2) return unit matrix, theta not relevant
        R = np.eye(2)
        V = np.eye(2)
    else:
        if theta is None:
            #  theta is not given, extract it
            theta = norm(w)
            w = unit(w)

        skw = skew(w)
        R = np.eye(2) + math.sin(theta) * skw + (1.0 - math.cos(theta)) * skw @ skw
        V = None
    
    if t is None:
        # so(2) case
        return R
    else:
        # se(3) case
        if V is None:
            V = np.eye(3) + (1.0-math.cos(theta))*skw/theta + (theta-math.sin(theta))/theta*skw @ skw
        return rt2tr(R, V@t)


def trprint(T, orient='rpy/zyx', label=None, file=sys.stdout, fmt='{:8.2g}', unit='deg'):
    """
    Compact display of SO(2) or SE(2) matrices
    
    :param T: matrix to format
    :type T: numpy.ndarray, shape=(2,2) or (3,3)
    :param label: text label to put at start of line
    :type label: str
    :param orient: 3-angle convention to use
    :type orient: str
    :param file: file to write formatted string to
    :type file: str
    :param fmt: conversion format for each number
    :type fmt: str
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: optional formatted string
    :rtype: str
    
    The matrix is formatted and written to ``file`` or if ``file=None`` then the
    string is returned. Orientation is expressed in one of several formats:
    - 'rpy/zyx' roll-pitch-yaw angles in ZYX axis order [default]
    - 'rpy/yxz' roll-pitch-yaw angles in YXZ axis order
    - 'rpy/zyx' roll-pitch-yaw angles in ZYX axis order
    - 'eul' Euler angles in ZYZ axis order
    - 'angvec' angle and axis
    
    - ``trprint(R)`` displays the SO(3) rotation matrix in a compact 
      single-line format:
        
        [LABEL:] ORIENTATION UNIT
        
    - ``trprint(T)`` displays the SE(3) homogoneous transform in a compact 
      single-line format:
        
        [LABEL:] [t=X, Y, Z;] ORIENTATION UNIT

    Example:
        
    >>> T = transl(1,2,3) @ rpy2tr(10, 20, 30, 'deg')
    >>> trprint(T, file=None, label='T')
    'T: t =        1,        2,        3; rpy/zyx =       10,       20,       30 deg'
    >>> trprint(T, file=None, label='T', orient='angvec')
    'T: t =        1,        2,        3; angvec = (      56 deg |     0.12,     0.62,     0.78)'
    >>> trprint(T, file=None, label='T', orient='angvec', fmt='{:8.4g}')
    'T: t =        1,        2,        3; angvec = (   56.04 deg |    0.124,   0.6156,   0.7782)'

    Notes:
        
     - If the 'rpy' option is selected, then the particular angle sequence can be
       specified with the options 'xyz' or 'yxz' which are passed through to ``tr2rpy``.
       'zyx' is the default.
      
    :seealso: trprint2, tr2eul, tr2rpy, tr2angvec
    """
    
    s = ''
    
    if label is not None:
        s += '{:s}: '.format(label)
    
    # print the translational part if it exists
    if ishom(T):
        s += 't = {};'.format(_vec2s(fmt, transl(T)))
    
    # print the angular part in various representations

    a = orient.split('/')
    if a[0] == 'rpy':
        if len(a) == 2:
            seq = a[1]
        else:
            seq = None
        angles = tr2rpy(t2r(T), order=seq, unit=unit)
        s += ' {} = {} {}'.format(orient, _vec2s(fmt, angles), unit)
        
    elif a[0].startswith('eul'):
        angles = tr2eul(t2r(T), unit)
        s += ' eul = {} {}'.format(_vec2s(fmt, angles), unit)
    
    elif a[0] == 'angvec':
        pass
        # as a vector and angle
        (theta,v) = tr2angvec(T, unit)
        if theta == 0:
            s += ' R = nil'
        else:
            s += ' angvec = ({} {} | {})'.format(fmt.format(theta), unit, _vec2s(fmt, v))
    else:
        raise ValueError('bad orientation format')
       
    if file:
        print(s, file=file)
    else:
        return s

    
def trprint2(T, label=None, file=sys.stdout, fmt='{:8.2g}', unit='deg'):
    """
    Compact display of SO(2) or SE(2) matrices
    
    :param T: matrix to format
    :type T: numpy.ndarray, shape=(2,2) or (3,3)
    :param label: text label to put at start of line
    :type label: str
    :param file: file to write formatted string to
    :type file: str
    :param fmt: conversion format for each number
    :type fmt: str
    :param unit: angular units: 'rad' [default], or 'deg'
    :type unit: str
    :return: optional formatted string
    :rtype: str
    
    The matrix is formatted and written to ``file`` or if ``file=None`` then the
    string is returned.
    
    - ``trprint2(R)`` displays the SO(2) rotation matrix in a compact 
      single-line format:
        
        [LABEL:] THETA UNIT
        
    - ``trprint2(T)`` displays the SE(2) homogoneous transform in a compact 
      single-line format:
        
        [LABEL:] [t=X, Y;] THETA UNIT

    Example:
        
    >>> T = transl2(1,2)@trot2(0.3)
    >>> trprint2(a, file=None, label='T')
    'T: t =        1,        2;       17 deg'

    :seealso: trprint
    """
    
    s = ''
    
    if label is not None:
        s += '{:s}: '.format(label)
    
    # print the translational part if it exists
    s += 't = {};'.format(_vec2s(fmt, transl2(T)))
    
    angle = math.atan2(T[1,0], T[0,0])
    if unit == 'deg':
        angle *= 180.0/math.pi
    s += ' {} {}'.format(_vec2s(fmt, [angle]), unit)
    
    if file:
        print(s, file=file)
    else:
        return s
    
def _vec2s(fmt, v):
        v = [x if np.abs(x) > 100*np.finfo(np.float64).eps else 0.0 for x in v ]
        return ', '.join([fmt.format(x) for x in v])
    
if __name__ == "main":
    a=transl2(1,2)@trot2(0.3)
    trprint2(a, file=None, label='T')
    
    T = transl(1,2,3) @ rpy2tr(10, 20, 30, 'deg')
    trprint(T, file=None, label='T')


