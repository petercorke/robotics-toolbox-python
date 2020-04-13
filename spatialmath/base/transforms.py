# Created by: Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan
# 1 June, 2017
""" This file contains all of the transforms functions that will be used within the toolbox"""

import sys
import math
import numpy as np
from scipy.linalg import expm
from spatialmath.base import argcheck

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

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: 3x3 rotation matrix
    :rtype: numpy array

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

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: 3x3 rotation matrix
    :rtype: numpy array

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

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: 3x3 rotation matrix
    :rtype: numpy array

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

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param t: the translation, defaults to [0,0,0]
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy array

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

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param t: the translation, defaults to [0,0,0]
    :return: 4x4 homogeneous transformation matrix as a numpy array

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

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param t: the translation, defaults to [0,0,0]
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy array

    - ``trotz(THETA) is a homogeneous transformation (4x4) representing a rotation
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
    :param y: translation along Y-axis
    :param z: translation along Z-axis
    :return: 4x4 homogeneous transformation matrix
    :rtype: numpy array

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

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :return: 2x2 rotation matrix
    :rtype: numpy array

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

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param t: the translation, defaults to [0,0]
    :return: 3x3 homogeneous transformation matrix
    :rtype: numpy array

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

    :param x: x translation, homogeneous transform or a list of translations
    :param y: y translation
    :return: homogeneous transform matrix or the translation elements of a homogeneous transform

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
    :rtype: numpy ndarray
    :raises ValueError: for zero length vector
    
    ``unit(v)`` returns a vector parallel to `v` of unit length.
    
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
    
    ``norm(v)`` is the 2-norm (length, magnitude) of the vector ``v``.
    
    :seealso: unit
    
    """
    return np.linalg.norm(v)

def isunit(v):
    return abs(np.linalg.norm(v)-1) < 100*np.finfo(np.float64).eps

# ---------------------------------------------------------------------------------------#
def r2t(R, check=False):
    """
    Convert SO(n) to SE(n)

    :param R: rotation matrix
    :param check: check if rotation matrix is valid (default False, no check)
    :return: homogeneous transformation matrix
    :rtype: numpy ndarray

    ``r2t(R)`` is an SE(2) or SE(3) homogeneous transform equivalent to an
    SO(2) or SO(3) orthonormal rotation matrix ``R`` with a zero translational
    component
    
    - if ``R`` is 2x2 then return is 3x3: SO(2) -> SE(2)
    - if ``R`` is 3x3 then return is 4x4: SO(3) -> SE(3)
    
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
    :rtype: numpy ndarray


    ``T2R(T)`` is the orthonormal rotation matrix component of homogeneous
    transformation matrix ``T``
    
    - if ``T`` is 3x3 then return is 2x2: SE(2) -> SO(2)
    - if ``T`` is 4x4 then return is 3x3: SE(3) -> SO(3)
    
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

    (R,t) = tr2rt(T) splits a homogeneous transformation matrix (NxN) into an orthonormal
    rotation matrix R (MxM) and a translation vector T (Mx1), where N=M+1.

    Works for `T` in SE(2) or SE(3):
        
    - If TR is 4x4, then R is 3x3 and T is 3x1.
    - If TR is 3x3, then R is 2x2 and T is 2x1.
    
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

    RT2TR(R, t) is a homogeneous transformation matrix (N+1xN+1) formed from an
    orthonormal rotation matrix R (NxN) and a translation vector t
    (Nx1).  Works for `R` in SO(2) or SO(3):
        
    - If R is 2x2 and t is 2x1, then TR is 3x3
    - If R is 3x3 and t is 3x1, then TR is 4x4
    
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

def isR(R):
    r"""
    Test if matrix is SO(n)
    
    :param R: matrix to test
    :return: whether matrix is a rotation matrix
    :rtype: bool
    
    Checks orthogonality, ie. :math:`{\bf R} {\bf R}^T = {\bf I}`
    
    :seealso: isrot2, isrot
    """
    #return np.abs(np.linalg.det(R) - 1) < 100*np.finfo(np.float64).eps
    return np.linalg.norm( R@R.T - np.eye(R.shape[0]) ) < 100*np.finfo(np.float64).eps

def ishom(T, check=False):
    """
    Test if matrix is an SE(3) homogeneous transformation
    
    :param T: matrix to test
    :param check: check validity of rotation submatrix
    :return: whether matrix belongs to SE(3)
    :rtype: bool
    
    - ``ISHOM(T)`` is True if the argument ``T`` is of dimension 4x4
    - ``ISHOM(T, check=True)`` as above, but also checks orthogonality of the rotation sub-matrix.
    
    :seealso: isrot, ishom2
    """
    return T.shape == (4,4) and (not check or isR(T[:3,:3]))

def ishom2(T, check=False):
    """
    Test if matrix is an SE(2) homogeneous transformation
    
    :param T: matrix to test
    :param check: check validity of rotation submatrix
    :return: whether matrix belongs to SE(2)
    :rtype: bool
    
    - ``ISHOM2(T)`` is True if the argument ``T`` is of dimension 3x3
    - ``ISHOM2(T, check=True)`` as above, but also checks orthogonality of the rotation sub-matrix.
    
    :seealso: isrot2, ishom, isvec
    """
    return T.shape == (3,3) and (not check or isR(T[:2,:2]))

def isrot(R, check=False):
    """
    Test if matrix is an SO(3) rotation matrix
    
    :param R: matrix to test
    :param check: check validity of rotation submatrix
    :return: whether matrix belongs to SO(3)
    :rtype: bool
    
    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 3x3
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.
    
    :seealso: isrot2, ishom
    """
    return R.shape == (3,3) and (not check or isR(R))

def isrot2(R, check=False):
    """
    Test if matrix is an SO(2) rotation matrix
    
    :param R: matrix to test
    :param check: check validity of rotation submatrix
    :return: whether matrix belongs to SO(2)
    :rtype: bool
    
    - ``ISROT(R)`` is True if the argument ``R`` is of dimension 2x2
    - ``ISROT(R, check=True)`` as above, but also checks orthogonality of the rotation matrix.
    
    :seealso: ishom2, isrot
    """
    return R.shape == (2,2) and (not check or isR(R))

#========================= angle sequences

# ---------------------------------------------------------------------------------------#
def rpy2r(roll, pitch=None, yaw=None, *, unit='rad', order='zyx'):
    """
    RPY2R Roll-pitch-yaw angles to rotation matrix

    :param theta: list of angles
    :param order: 'xyz', 'zyx' or 'yxz'
    :param unit: 'rad' or 'deg'
    :return: rotation matrix

    RPY2R(ROLL, PITCH, YAW, OPTIONS) is an SO(3) orthonormal rotation matrix
    (3x3) equivalent to the specified roll, pitch, yaw angles angles.
    These correspond to rotations about the Z, Y, X axes respectively. If ROLL,
    PITCH, YAW are column vectors (Nx1) then they are assumed to represent a
    trajectory and R is a three-dimensional matrix (3x3xN), where the last index
    corresponds to rows of ROLL, PITCH, YAW.

    R = RPY2R(RPY, OPTIONS) as above but the roll, pitch, yaw angles are taken
    from the vector (1x3) RPY=[ROLL,PITCH,YAW]. If RPY is a matrix(Nx3) then R
    is a three-dimensional matrix (3x3xN), where the last index corresponds to
    rows of RPY which are assumed to be [ROLL,PITCH,YAW].

    Options::
        'deg'   Compute angles in degrees (radians default)
        'xyz'   Rotations about X, Y, Z axes (for a robot gripper)
        'yxz'   Rotations about Y, X, Z axes (for a camera)

    Note::
    - Toolbox rel 8-9 has the reverse angle sequence as default.
    - ZYX order is appropriate for vehicles with direction of travel in the X
    direction.  XYZ order is appropriate if direction of travel is in the Z direction.
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
def rpy2tr(roll, pitch=None, yaw=None, *, order='zyx', unit='rad'):
    """
    RPY2TR Roll-pitch-yaw angles to homogeneous transform

    :param thetas: list of angles
    :param order: order can be 'xyz'/'arm', 'zyx'/'vehicle', 'yxz'/'camera'
    :param unit: unit of input angles
    :return: homogeneous transformation matrix

    T = RPY2TR(ROLL, PITCH, YAW, OPTIONS) is an SE(3) homogeneous
    transformation matrix (4x4) with zero translation and rotation equivalent
    to the specified roll, pitch, yaw angles angles. These correspond to
    rotations about the Z, Y, X axes respectively. If ROLL, PITCH, YAW are
    column vectors (Nx1) then they are assumed to represent a trajectory and
    R is a three-dimensional matrix (4x4xN), where the last index corresponds
    to rows of ROLL, PITCH, YAW.

    T = RPY2TR(RPY, OPTIONS) as above but the roll, pitch, yaw angles are
    taken from the vector (1x3) RPY=[ROLL,PITCH,YAW]. If RPY is a matrix
    (Nx3) then R is a three-dimensional matrix (4x4xN), where the last index
    corresponds to rows of RPY which are assumed to be
    ROLL,PITCH,YAW].

    Options::
     'deg'   Compute angles in degrees (radians default)
     'xyz'   Rotations about X, Y, Z axes (for a robot gripper)
     'yxz'   Rotations about Y, X, Z axes (for a camera)

    Note::
    - Toolbox rel 8-9 has the reverse angle sequence as default.
    - ZYX order is appropriate for vehicles with direction of travel in the X
    direction.  XYZ order is appropriate if direction of travel is in the Z
    direction.
    """

    R = rpy2r(roll, pitch, yaw, order=order, unit=unit)
    return r2t(R)

# ---------------------------------------------------------------------------------------#
def eul2r(phi, theta=None, psi=None, *, unit='rad'):
    """
    EUL2R Convert Euler angles to rotation matrix

    :param phi: x axis rotation
    :param theta: y axis rotation
    :param psi: z axis rotation
    :param unit: 'rad' or 'deg' for angles
    :return: rotation matrix

    R = EUL2R(PHI, THETA, PSI, UNIT) is an SO(3) orthonornal rotation
    matrix (3x3) equivalent to the specified Euler angles.  These correspond
    to rotations about the Z, Y, Z axes respectively. If PHI, THETA, PSI are
    column vectors (Nx1) then they are assumed to represent a trajectory and
    R is a three-dimensional matrix (3x3xN), where the last index corresponds
    to rows of PHI, THETA, PSI.

    R = EUL2R(EUL, OPTIONS) as above but the Euler angles are taken from the
    vector (1x3)  EUL = [PHI THETA PSI]. If EUL is a matrix (Nx3) then R is a
    three-dimensional matrix (3x3xN), where the last index corresponds to
    rows of RPY which are assumed to be [PHI,THETA,PSI].

    Options::
    'deg'      Angles given in degrees (radians default)

    Note::
    - The vectors PHI, THETA, PSI must be of the same length.
    """
    
    if np.isscalar(phi):
        angles = [phi, theta, psi]
    else:
        angles = argcheck.getvector(phi, 3)
        
    angles = argcheck.getunit(angles, unit)
        
    return rotz(angles[0]) @ roty(angles[1]) @ rotz(angles[2])


# ---------------------------------------------------------------------------------------#
def eul2tr(phi, theta=None, psi=None, *, unit='rad'):
    """
    EUL2TR Convert Euler angles to homogeneous transform

    :param phi: x axis rotation
    :param theta: y axis rotation
    :param psi: z axis rotation
    :param unit: 'rad' or 'deg' for angles
    :return: rotation matrix

    T = EUL2TR(PHI, THETA, PSI, OPTIONS) is an SE(3) homogeneous
    transformation matrix (4x4) with zero translation and rotation equivalent
    to the specified Euler angles. These correspond to rotations about the Z,
    Y, Z axes respectively. If PHI, THETA, PSI are column vectors (Nx1) then
    they are assumed to represent a trajectory and R is a three-dimensional
    matrix (4x4xN), where the last index corresponds to rows of PHI, THETA,
    PSI.

    R = EUL2R(EUL, OPTIONS) as above but the Euler angles are taken from the
    vector (1x3)  EUL = [PHI THETA PSI]. If EUL is a matrix (Nx3) then R is a
    three-dimensional matrix (4x4xN), where the last index corresponds to
    rows of RPY which are assumed to be [PHI,THETA,PSI].

    Options::
    'deg'      Angles given in degrees (radians default)

    Note::
    - The vectors PHI, THETA, PSI must be of the same length.
    - The translational part is zero.
    """
    
    R = eul2r(phi, theta, psi, unit=unit)
    return r2t(R)

# ---------------------------------------------------------------------------------------#
def angvec2r(theta, v, *, unit='rad'):
    """
    ANGVEC2R(THETA, V) is an orthonormal rotation matrix (3x3)
    equivalent to a rotation of THETA about the vector V.

    :param theta: rotation in radians
    :param v: vector
    :return: rotation matrix

    Notes::
    - If THETA == 0 then return identity matrix.
    - If THETA ~= 0 then V must have a finite length.
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
    ANGVEC2TR(THETA, V) is a homogeneous transform matrix (4x4) equivalent to a
    rotation of THETA about the vector V.

    :param theta: rotation in radians
    :param v: vector
    :return: homogenous transform matrix

    Notes::
    - The translational part is zero.
    - If THETA == 0 then return identity matrix.
    - If THETA ~= 0 then V must have a finite length.
    """
    return r2t(angvec2r(theta, v, unit=unit))


# ---------------------------------------------------------------------------------------#
def oa2r(o, a=None):
    """
    Define SO(3) rotation from two vectors

    :param o: vector parallel to Y- axes
    :type o: list, tuple, numpy.ndarray
    :param a: vector parallel to the z-axes
    :type o: list, tuple, numpy.ndarray
    :return: SO(3)  rotation matrix
    :rtype: numpy.ndarray, shape=(3,3)

    ``T = oa2tr(O, A)`` is an SO(3) homogeneous tranformation (3,3) for the
    specified orientation and approach vectors (3x1) formed from 3 vectors
    such that R = [N O A] and N = O x A.

    Notes:
        
    - The rotation submatrix is guaranteed to be orthonormal so long as O and A
      are not parallel.
    - The translational part is zero.
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
    Define SE(3) rotation from two vectors

    :param o: vector parallel to Y- axes
    :type o: list, tuple, numpy.ndarray
    :param a: vector parallel to the z-axes
    :type o: list, tuple, numpy.ndarray
    :return: SE(3) pure rotation matrix
    :rtype: numpy.ndarray, shape=(4,4)

    ``T = oa2tr(O, A)`` is an SE(3) homogeneous tranformation (4x4) for the
    specified orientation and approach vectors (3x1) formed from 3 vectors
    such that R = [N O A] and N = O x A.

    Notes:
        
    - The rotation submatrix is guaranteed to be orthonormal so long as O and A
      are not parallel.
    - The translational part is zero.
    - The vectors O and A are parallel to the Y- and Z-axes of the equivalent coordinate frame.
    """
    return r2t(oa2r(o, a))


# ------------------------------------------------------------------------------------------------------------------- #
def tr2angvec(R, unit='rad'):
    r"""
    Convert SO(3) or SE(3) to angle and rotation vector
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :return: :math:`(\theta, {\bf v})`
    :rtype: float, numpy.ndarray, shape=(3,)
    
    ``tr2angvec(R)`` is a rotation angle and a vector about which the rotation
    acts that corresponds to the rotation part of ``R``. 
        
    By default the angle is in radians but can be changed setting `unit='deg'`.
    
    Notes:
        
    - If the input is SE(3) the translation component is ignored.
    
    :seealso: angvec2r, angvec2tr, tr2rpy, tr2eul
    
    """
    check_args.unit_check(unit)
    check_args.tr2angvec(tr=tr, unit=unit)

    if common.isrot(tr) is False:
        tr = t2r(tr)

    if common.isrot(tr) or common.ishomog(tr, dim=[4, 4]):
        if tr.ndim > 2:
            theta = np.zeros([tr.shape[2], 1])
            n = np.zeros([tr.shape[2], 3])
        else:
            theta = np.zeros([1, 1])
            n = np.zeros([1, 3])

        for i in range(0, theta.shape[0]):
            if theta.shape[0] == 1:
                tri = tr[:, :]
            else:
                tri = tr[i, :, :]

            if abs(np.linalg.det(tri) - 1) < 10 * np.spacing([1])[0]:
                [th, v] = trlog(tri)
                theta[i, 0] = th
                n[i, :] = v
                if unit == 'deg':
                    theta[i, 0] = theta[i, 0] * 180 / math.pi
                print('Rotation: ', theta[i, 0], unit, 'x', '[', n[i, :], ']')
            else:
                raise TypeError('Matrix in not orthonormal.')
    else:
        raise TypeError('Argument must be a SO(3) or SE(3) matrix.')


# ------------------------------------------------------------------------------------------------------------------- #
def tr2eul(R, unit='rad', flip=False):
    r"""
    Convert SO(3) or SE(3) to ZYX Euler angles
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
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
    
    # check_args.unit_check(unit)
    # check_args.tr2eul(tr=tr, unit=unit, flip=flip)

    # if tr.ndim > 2:
    #     eul = np.zeros([tr.shape[2], 3])
    #     for i in range(0, tr.shape[2]):
    #         eul[i, :] = tr2eul(tr[i, :, :])
    #     return eul
    # else:
    #     eul = np.zeros((3,))

    # if abs(tr[0, 2]) < np.spacing([1])[0] and abs(tr[1, 2]) < np.spacing([1])[0]:
    #     eul[0] = 0
    #     sp = 0
    #     cp = 0
    #     eul[1] = math.atan2(cp * tr[0, 2] + sp * tr[1, 2], tr[2, 2])
    #     eul[2] = math.atan2(-sp * tr[0, 0] + cp * tr[1, 0], -sp * tr[0, 1] + cp * tr[1, 1])
    # else:
    #     if flip:
    #         eul[0] = math.atan2(-tr[1, 2], -tr[0, 2])
    #     else:
    #         eul[0] = math.atan2(tr[1, 2], tr[0, 2])
    #     sp = math.sin(eul[0])
    #     cp = math.cos(eul[0])
    #     eul[0] = math.atan2(cp * tr[0, 2] + sp * tr[1, 2], tr[2, 2])
    #     eul[2] = math.atan2(-sp * tr[0, 0] + cp * tr[1, 0], -sp * tr[0, 1] + cp * tr[1, 1])

    # if unit == 'deg':
    #     eul = eul * 180 / math.pi

    # return eul
    pass


# ------------------------------------------------------------------------------------------------------------------- #
def tr2rpy(R, unit='rad', order='zyx', check=False):
    r"""
    Convert SO(3) or SE(3) to roll-pitch-yaw angles
    
    :param R: SO(3) or SE(3) matrix
    :type R: numpy.ndarray, shape=(3,3) or (4,4)
    :param unit: 'rad' or 'deg'
    :type unit: str
    :param order: 'xyz', 'zyx' or 'yxz' [default 'zyx']
    :type unit: str
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
    # check_args.unit_check(unit)
    # check_args.tr2rpy(tr=tr, unit=unit, order=order)
    
    
    # if check and np.abs(np.linalg.det(R) - 1) < 100*np.finfo(np.float64).eps:
    #         raise ValueError('Invalid rotation matrix')

    # if tr.ndim > 2:
    #     rpy = np.zeros((tr.shape[2], 3))
    #     for i in range(0, tr.shape[2]):
    #         rpy[i, :] = tr2rpy(tr[i, :, :])
    #     return rpy
    # else:
    #     rpy = np.zeros((3,))

    # if common.isrot(tr) or common.ishomog(tr, dim=[4, 4]):
    #     if order == 'xyz' or order == 'arm':
    #         if abs(abs(tr[0, 2]) - 1) < np.spacing([1])[0]:
    #             rpy[0] = 0
    #             rpy[1] = math.asin(tr[0, 2])
    #             if tr[2] > 0:
    #                 rpy[2] = math.atan2(tr[2, 1], tr[1, 1])
    #             else:
    #                 rpy[2] = -math.atan2(tr[1, 0], tr[2, 0])
    #         else:
    #             rpy[0] = -math.atan2(tr[0, 1], tr[0, 0])
    #             rpy[1] = math.atan2(tr[0, 2] * math.cos(rpy[0, 0]), tr[0, 0])
    #             rpy[2] = -math.atan2(tr[1, 2], tr[2, 2])
    #     if order == 'zyx' or order == 'vehicle':
    #         if abs(abs(tr[2, 0]) - 1) < np.spacing([1])[0]:
    #             rpy[0] = 0
    #             rpy[1] = -math.asin(tr[2, 0])
    #             if tr[2, 0] < 0:
    #                 rpy[2] = -math.atan2(tr[0, 1], tr[0, 2])
    #             else:
    #                 rpy[2] = math.atan2(-tr[0, 1], -tr[0, 2])
    #         else:
    #             rpy[0] = math.atan2(tr[2, 1], tr[2, 2])
    #             rpy[1] = math.atan2(-tr[2, 0] * math.cos(rpy[0]), tr[2, 2])
    #             rpy[2] = math.atan2(tr[1, 0], tr[0, 0])
    #     if order == 'yxz' or order == 'camera':
    #         if abs(abs(tr[1, 2]) - 1) < np.spacing([1])[0]:
    #             rpy[0] = 0
    #             rpy[1] = -math.asin(tr[1, 2])
    #             if tr[1, 2] < 0:
    #                 rpy[2] = -math.atan2(tr[2, 0], tr[0, 0])
    #             else:
    #                 rpy[2] = math.atan2(-tr[2, 0], -tr[2, 1])
    #         else:
    #             rpy[0] = math.atan2(tr[1, 0], tr[1, 1])
    #             rpy[1] = math.atan2(-math.cos(rpy[0]) * tr[1, 2], tr[1, 1])
    #             rpy[2] = math.atan2(tr[0, 2], tr[2, 2])
    # else:
    #     raise TypeError('Argument must be a 3x3 or 4x4 matrix.')

    # if unit == 'deg':
    #     rpy = rpy * 180 / math.pi

    # return rpy
    pass

# ---------------------------------------------------------------------------------------#
def skew(v):
    r"""
    SKEW Skew-symmetric metrix from vector

    :param v: 1 or 3 vector
    :return: skew-symmetric matrix in so(2) or so(3)
    :rtype: numpy ndarray
    
    ``skew(V)`` is a skew-symmetric matrix formed from V.

    If V (1x1) then S = :math:`\left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]`
        

    and if V (1x3) then S = :math:`\left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]`


    Notes:
        
    - This is the inverse of the function ``vex()``.
    - These are the generator matrices for the Lie algebras so(2) and so(3).
    
    :seealso: vex
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
    VEX Skew-symmetric matrix to vector

    :param s: skew-symmetric matrix
    :return: vector of unique values
    :rtype: numpy ndarray

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.
    
    For the so(2) case (2x2 matrix) where ``S`` :math:`= \left[ \begin{array}{cc} 0 & -v \\ v & 0 \end{array} \right]` then ``V`` is 1x1
    
    For the so(3) case (3x3 matrix) where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0\end{array} \right]` then ``V`` is 3x1.
    
    Notes:
        
    - This is the inverse of the function SKEW().
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
      is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
      element of the matrix.
      
    :seealso: skew
    """
    if s.shape == (3,3):
        return 0.5 * np.array([s[2, 1] - s[1, 2], s[0, 2] - s[2, 0], s[1, 0] - s[0, 1]])
    elif s.shape == (2, 2):
        return 0.5 * np.array([s[1, 0] - s[0, 1]])
    else:
        raise AttributeError("Argument must be 2x2 or 3x3 matrix")

# ---------------------------------------------------------------------------------------#
def skewa(v):
    r"""
    Augmented skew-symmetric metrix from vector

    :param v: 1 or 3 vector
    :return: augmented skew-symmetric matrix in se(2) or se(3)
    :rtype: numpy ndarray
    
    ``skewa(V)`` is an augmented skew-symmetric matrix formed from V.

    If V (1x1) then S = :math:`\left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]`
        

    and if V (1x3) then S = :math:`\left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]`

    Notes:
        
    - This is the inverse of the function ``vexa()``.
    - These are the generator matrices for the Lie algebras se(2) and se(3).
    - Map twist vectors in 2D and 3D space to se(2) and se(3).
    
    :seealso: vexa
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
    VEX Skew-symmetric matrix to vector

    :param s: augmented skew-symmetric matrix
    :return: vector of unique values
    :rtype: numpy ndarray

    ``vex(S)`` is the vector which has the corresponding skew-symmetric matrix ``S``.
    
    For the se(2) case (3x3 matrix) where ``S`` :math:`= \left[ \begin{array}{ccc} 0 & -v_3 & v_1 \\ v_3 & 0 & v_2 \\ 0 & 0 & 0 \end{array} \right]` then ``V`` is 3x1
    
    For the se(3) case (4x4 matrix) where ``S`` :math:`= \left[ \begin{array}{cccc} 0 & -v_6 & v_5 & v_1 \\ v_6 & 0 & -v_4 & v_2 \\ -v_5 & v_4 & 0 & v_3 \\ 0 & 0 & 0 & 0 \end{array} \right]` then ``V`` is 6x1.
    
    Notes:
        
    - This is the inverse of the function SKEW().
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
      is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
      element of the matrix.
      
    :seealso: skewa
    """
    if Omega.shape == (4,4):
        return np.hstack( (transl(Omega), vex(t2r(Omega))) )
    elif Omega.shape == (3,3):
        return np.hstack( (transl2(Omega), vex(t2r(Omega))) )
    else:
        raise AttributeError("expecting a 3x3 or 4x4 matrix")
        


# ---------------------------------------------------------------------------------------#
def trlog(T):
    """
    TRLOG logarithm of SO(3) or SE(3) matrix

    :param T: SO(3) or SE(3) Matrix
    :return: [rotation, vector]

    [theta,w] = trlog(R) as above but returns directly theta the rotation angle and w
    (3x1) the unit-vector indicating the rotation axis.

    [theta,twist] = trlog(T) as above but returns directly theta the rotation angle
    and a twist vector (6x1) comprising [v w].

    Notes::
    - Efficient closed-form solution of the matrix logarithm for arguments that are
    SO(3) or SE(3).
    - Special cases of rotation by odd multiples of pi are handled.
    - Angle is always in the interval [0,pi].
    """
    tr = T.trace()
    if common.isrot(T):
        # deal with rotation matrix
        R = T
        if abs(tr[0, 0] - 3) < 100 * np.spacing([1])[0]:
            # matrix is identity
            w = [0, 0, 0]
            theta = 0
        elif abs(tr[0, 0] + 1) < 100 * np.spacing([1])[0]:
            # tr R = -1
            # rotation by +/- po, +/- 3pi ect
            mx = R.diagonal().max()
            k = R.diagonal().argmax()
            I = np.eye(3)
            col = R[:, k] + I[:, k]
            w = col / np.sqrt(2 * (1 + mx))
            theta = np.pi
        else:
            # general case
            theta = np.arccos((tr[0, 0] - 1) / 2)
            skw = (R - R.conj().transpose()) / 2 / np.sin(theta)
            w = vex(skw)
        return [theta, w]
    elif common.ishomog(T, [T.shape[0], T.shape[1]]):
        # SE(3) matrix
        [R, t] = tr2rt(T)

        if (tr - 3) < 100 * np.spacing([1])[0]:
            # is identity matrix
            w = np.matrix([[0, 0, 0]])
            v = t
            theta = 1
            skw = np.zeros(3)
        else:
            [theta, w] = trlog(R)
            skw = skew(w)

        Ginv = np.eye(3) / theta - skw / 2 + (1 / theta - 1 / np.tan(theta / 2) / 2) * skw ** 2
        v = Ginv * t
        return [theta, w]
    else:
        raise AttributeError("Expect SO(3) or SE(3) matrix")

# ---------------------------------------------------------------------------------------#
def trexp(S, theta=None):
    """
    TREXP matrix exponential for so(3) and se(3)

    :param S: SO(3), SE(3), unit vector or twist vector
    :param theta: Rotation in radians
    :return: matrix exponential
    For so(3)::

    TREXP(OMEGA) is the matrix exponential (3x3) of the so(3) element OMEGA that
    yields a rotation matrix (3x3).

    TREXP(OMEGA, THETA) as above, but so(3) motion of THETA*OMEGA.

    TREXP(S, THETA) as above, but rotation of THETA about the unit vector S.

    TREXP(W) as above, but the so(3) value is expressed as a vector W (1x3) where
    W = S * THETA. Rotation by ||W|| about the vector W.

    For se(3)::

    TREXP(SIGMA) is the matrix exponential (4x4) of the se(3) element SIGMA that
    yields a homogeneous transformation  matrix (4x4).

    TREXP(TW) as above, but the se(3) value is expressed as a twist vector TW (1x6).

    TREXP(SIGMA, THETA) as above, but se(3) motion of SIGMA*THETA, the rotation
    part of SIGMA (4x4) must be unit norm.

    TREXP(TW, THETA) as above, but se(3) motion of TW*THETA, the rotation part
    of TW (1x6) must be unit norm.

    Notes::
    - Efficient closed-form solution of the matrix exponential for arguments
    that are so(3) or se(3).
    - If theta is given then the first argument must be a unit vector or a
    skew-symmetric matrix from a unit vector.
    - Angle vector argument order is different to ANGVEC2R.
    """
    if argcheck.ismatrix(S, (4,4)) or argcheck.isvector(S, 6):
        # se(3) case
        if argcheck.isvector(S, 6):
            S = skewa(S)
            
        if not theta:
            return np.linalg.exp(S)
    
        else:
            # se(3) with a twist angle specified
            [skw, v] = tr2rt(S)
            
            R = trexp(skw, theta)
            t = (np.eye(3) + np.sin(theta) * skw + (1 - np.cos(theta)) * skw @ skw) @ v
            return rt2tr(R, t)
        
    elif argcheck.ismatrix(S, (3,3)) or argcheck.isvector(S, 3):
        if argcheck.ismatrix(S, (3,3)):
            w = vex(S)
        else:
            w = argcheck.getvector(S, 3, 'array')
        
        if np.linalg.norm(w) < 10 * np.finfo(np.float64).eps:
            # for a zero so(3) return unit matrix, theta not relevant
            return np.eye(3)
        
        if theta:
            assert isunit(np.linalg.norm(w)), 'w must be a unit twist'
        else:
            #  theta is not given, extract it
            theta = np.linalg.norm(w)
            w = unit(w)
        print(w)
        S = skew(w)
        print(S)
        return np.eye(3) + np.sin(theta) * S + (1 - np.cos(theta)) * S @ S

    else:
        raise AttributeError(" First argument must be SO(3), 3-vector, SE(3) or 6-vector")



# ---------------------------------------------------------------------------------------#
def trexp2(S, theta=None):
    """
    TREXP2 matrix exponential for so(2) and se(2)

    :param S:S: SO(2), SE(2) or unit vector
    :param theta:
    :return: matrix exponential

    R = TREXP2(OMEGA) is the matrix exponential (2x2) of the so(2) element OMEGA that
    yields a rotation matrix (2x2).

    R = TREXP2(THETA) as above, but rotation by THETA (1x1).

    SE(2)::

    T = TREXP2(SIGMA) is the matrix exponential (3x3) of the se(2) element
    SIGMA that yields a homogeneous transformation  matrix (3x3).

    T = TREXP2(TW) as above, but the se(2) value is expressed as a vector TW
    (1x3).

    T = TREXP2(SIGMA, THETA) as above, but se(2) rotation of SIGMA*THETA, the
    rotation part of SIGMA (3x3) must be unit norm.

    T = TREXP(TW, THETA) as above, but se(2) rotation of TW*THETA, the
    rotation part of TW must be unit norm.

    Notes::
    - Efficient closed-form solution of the matrix exponential for arguments that are
      so(2) or se(2).
    - If theta is given then the first argument must be a unit vector or a
      skew-symmetric matrix from a unit vector.
    """
    if common.ishomog(S, [3, 3]) or common.isvec(S, 3):
        if theta is None:
            if common.isvec(S, 3):
                S = skewa(S)  # todo
            return expm(S)
        else:
            if common.ishomog(S, [3, 3]):
                v = S[:2, 2]
                skw = S[:2, :2]
            else:
                v = S[:1, :2]
                skw = skew(S[2])
            R = trexp2(skw, theta)

            t = (np.eye(2) * theta + (1 - np.cos(theta)) * skw + (theta - np.sin(theta)) * skw * skw) * v
            return rt2tr(R, t)
    else:
        if common.isrot2(S):
            w = vex(S)
        elif common.isvec(S, 1):
            w = S
        else:
            raise AttributeError("Expecting scalar or 2x2")
        if theta is None:
            if np.linalg.norm(w) < 10 * np.spacing([1])[0]:
                return np.eye(2)
            theta = np.linalg.norm(w)
            S = skew(unitize(w))
        else:
            if theta < 10 * np.spacing([1])[0]:
                return np.eye(2)
            # todo isunit
            S = skew(w)
        return np.eye(2, 2) + np.sin(theta) * S + (1 - np.cos(theta)) * S ** 2


def trprint(T, orient='rpy/zyx', label=None, file=sys.stdout, fmt='%8.2g', unit='deg'):
    """
TRPRINT Compact display of SE(3) homogeneous transformation

 TRPRINT(T, OPTIONS) displays the homogoneous transform (4x4) in a compact 
 single-line format.  If T is a homogeneous transform sequence then each 
 element is printed on a separate line.

 TRPRINT(R, OPTIONS) as above but displays the SO(3) rotation matrix (3x3).

 S = TRPRINT(T, OPTIONS) as above but returns the string.

 TRPRINT T OPTIONS is the command line form of above.


 Options::
 'rpy'        display with rotation in ZYX roll/pitch/yaw angles (default)
 'xyz'        change RPY angle sequence to XYZ
 'yxz'        change RPY angle sequence to YXZ
 'euler'      display with rotation in ZYZ Euler angles
 'angvec'     display with rotation in angle/vector format
 'radian'     display angle in radians (default is degrees)
 'fmt', f     use format string f for all numbers, (default #g)
 'label',l    display the text before the transform
 'fid',F      send text to the file with file identifier F

 Examples::
        >> trprint(T2)
        t = (0,0,0), RPY/zyx = (-122.704,65.4084,-8.11266) deg

        >> trprint(T1, 'label', 'A')
               A:t = (0,0,0), RPY/zyx = (-0,0,-0) deg

       >> trprint B euler
       t = (0.629, 0.812, -0.746), EUL = (125, 71.5, 85.7) deg

 Notes::
 - If the 'rpy' option is selected, then the particular angle sequence can be
   specified with the options 'xyz' or 'yxz' which are passed through to TR2RPY.
  'zyx' is the default.

 See also TR2EUL, TR2RPY, TR2ANGVEC.
"""
    

    # orient rpy, rpy/xyz, rpy/zyx, rpy/yzx, eul
    
    s = ''
    
    if label is not None:
        s += '{:<8s}: '.format(label)
    
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
        s += ' eul = {:s} {:s}'.format(_vec2s(fmt, angles), unit)
    
    elif a[0] == 'angvec':
        pass
        # as a vector and angle
#        [th,v] = tr2angvec(T, unit):
#        if th == 0:
#            s = strcat(s, sprintf(' R = nil') );
#        elif opt.radian
#                s = strcat(s, sprintf(' R = (%srad | %s)', ...
#                    sprintf(opt.fmt, th), vec2s(opt.fmt, v)) );
#            else
#                s = strcat(s, sprintf(' R = (%sdeg | %s)', ...
#                    sprintf(opt.fmt, th*180.0/pi), vec2s(opt.fmt,v)) );
#            end
    else:
        raise ValueError('bad orientation format')
       
    if file:
        print(s, file=file)
    return s

    



def trprint2(T, label=None, file=sys.stdout, fmt='{:8.2g}', unit='deg'):
    """
TRPRINT2 Compact display of SE(2) homogeneous transformation

 TRPRINT2(T, OPTIONS) displays the homogoneous transform (3x3) in a compact 
 single-line format.  If T is a homogeneous transform sequence then each 
 element is printed on a separate line.

 TRPRINT2(R, OPTIONS) as above but displays the SO(2) rotation matrix (3x3).

 S = TRPRINT2(T, OPTIONS) as above but returns the string.

 TRPRINT2 T  is the command line form of above, and displays in RPY format.

 Options::
 'radian'     display angle in radians (default is degrees)
 'fmt', f     use format string f for all numbers, (default #g)
 'label',l    display the text before the transform

 Examples::
        >> trprint2(T2)
        t = (0,0), theta = -122.704 deg


 See also TRPRINT.
"""
    
    s = ''
    
    if label is not None:
        s += '{:<8s}: '.format(label)
    
    # print the translational part if it exists
    s += 't = {};'.format(_vec2s(fmt, transl2(T)))
    
    angle = math.atan2(T[1,0], T[0,0])
    s += ' {} {}'.format(_vec2s(fmt, [angle]), unit)
    
    if file:
        print(s, file=file)
    return s
    
def _vec2s(fmt, v):
        v = [x if np.abs(x) > 100*np.finfo(np.float64).eps else 0.0 for x in v ]
        return ', '.join([fmt % x for x in v])
    

