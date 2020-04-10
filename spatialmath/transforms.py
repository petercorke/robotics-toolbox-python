# Created by: Josh Carrigg Hodson, Aditya Dua, Chee Ho Chan
# 1 June, 2017
""" This file contains all of the transforms functions that will be used within the toolbox"""
import math
import numpy as np
from scipy.linalg import expm
from . import check_args
from ..tests import test_transforms
from . import common
import unittest
import vtk


# ---------------------------------------------------------------------------------------#
def rotx(theta, unit="rad"):
    """
    ROTX gives rotation about X axis

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: rotation matrix

    rotx(THETA) is an SO(3) rotation matrix (3x3) representing a rotation
    of THETA radians about the x-axis
    rotx(THETA, "deg") as above but THETA is in degrees
    """
    check_args.unit_check(unit)
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def roty(theta, unit="rad"):
    """
    ROTY Rotation about Y axis

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: rotation matrix

    roty(THETA) is an SO(3) rotation matrix (3x3) representing a rotation
    of THETA radians about the y-axis
    roty(THETA, "deg") as above but THETA is in degrees
    """
    check_args.unit_check(unit)
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def rotz(theta, unit="rad"):
    """
    ROTZ Rotation about Z axis

    :param theta: angle for rotation matrix
    :param unit: unit of input passed. 'rad' or 'deg'
    :return: rotation matrix

    rotz(THETA) is an SO(3) rotation matrix (3x3) representing a rotation
    of THETA radians about the z-axis
    rotz(THETA, "deg") as above but THETA is in degrees
    """
    check_args.unit_check(unit)
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def trotx(theta, unit="rad", xyz=[0, 0, 0]):
    """
    TROTX Rotation about X axis

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param xyz: the xyz translation, if blank defaults to [0,0,0]
    :return: homogeneous transform matrix

    trotx(THETA) is a homogeneous transformation (4x4) representing a rotation
    of THETA radians about the x-axis.
    trotx(THETA, 'deg') as above but THETA is in degrees
    trotx(THETA, 'rad', [x,y,z]) as above with translation of [x,y,z]
    """
    check_args.unit_check(unit)
    tm = rotx(theta, unit)
    tm = np.r_[tm, np.zeros((1, 3))]
    mat = np.c_[tm, np.array([[xyz[0]], [xyz[1]], [xyz[2]], [1]])]
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def troty(theta, unit="rad", xyz=[0, 0, 0]):
    """
    TROTY Rotation about Y axis

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param xyz: the xyz translation, if blank defaults to [0,0,0]
    :return: homogeneous transform matrix

    troty(THETA) is a homogeneous transformation (4x4) representing a rotation
    of THETA radians about the y-axis.
    troty(THETA, 'deg') as above but THETA is in degrees
    troty(THETA, 'rad', [x,y,z]) as above with translation of [x,y,z]
    """
    check_args.unit_check(unit)
    tm = roty(theta, unit)
    tm = np.r_[tm, np.zeros((1, 3))]
    mat = np.c_[tm, np.array([[xyz[0]], [xyz[1]], [xyz[2]], [1]])]
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def trotz(theta, unit="rad", xyz=[0, 0, 0]):
    """
    TROTZ Rotation about Z axis

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :param xyz: the xyz translation, if blank defaults to [0,0,0]
    :return: homogeneous transform matrix

    trotz(THETA) is a homogeneous transformation (4x4) representing a rotation
    of THETA radians about the z-axis.
    trotz(THETA, 'deg') as above but THETA is in degrees
    trotz(THETA, 'rad', [x,y,z]) as above with translation of [x,y,z]
    """
    check_args.unit_check(unit)
    tm = rotz(theta, unit)
    tm = np.r_[tm, np.zeros((1, 3))]
    mat = np.c_[tm, np.array([[xyz[0]], [xyz[1]], [xyz[2]], [1]])]
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def r2t(rmat):
    """
    R2T Convert rotation matrix to a homogeneous transform

    :param rmat: rotation matrix
    :return: homogeneous transformation

    R2T(rmat) is an SE(2) or SE(3) homogeneous transform equivalent to an
    SO(2) or SO(3) orthonormal rotation matrix rmat with a zero translational
    component. Works for T in either SE(2) or SE(3):
    if rmat is 2x2 then return is 3x3, or
    if rmat is 3x3 then return is 4x4.

    Translational component is zero.
    """
    assert isinstance(rmat, np.matrix)
    dim = rmat.shape
    if dim[0] != dim[1]:
        raise ValueError(' Matrix Must be square ')
    elif dim[0] == 2:
        tmp = np.r_[rmat, np.zeros((1, 2))]
        mat = np.c_[tmp, np.array([[0], [0], [1]])]
        mat = np.asmatrix(mat.round(15))
        return mat
    elif dim[0] == 3:
        tmp = np.r_[rmat, np.zeros((1, 3))]
        mat = np.c_[tmp, np.array([[0], [0], [0], [1]])]
        mat = np.asmatrix(mat.round(15))
        return mat
    else:
        raise ValueError(' Value must be a rotation matrix ')


# ---------------------------------------------------------------------------------------#
def t2r(tmat):
    """
    R2T Convert homogeneous transform to a rotation matrix

    :param tmat: homogeneous transform
    :return: rotation matrix

    T2R(tmat) is the orthonormal rotation matrix component of homogeneous
    transformation matrix tmat.  Works for T in SE(2) or SE(3)
    if tmat is 3x3 then return is 2x2, or
    if tmat is 4x4 then return is 3x3.

    Validity of rotational part is not checked
    """
    assert isinstance(tmat, np.matrix)
    dim = tmat.shape
    if dim[0] != dim[1]:
        raise ValueError(' Matrix Must be square ')
    elif dim[0] == 3:
        tmp = np.delete(tmat, [2], axis=0)
        mat = np.delete(tmp, [2], axis=1)
        mat = np.asmatrix(mat.round(15))
        return mat
    elif dim[0] == 4:
        tmp = np.delete(tmat, [3], axis=0)
        mat = np.delete(tmp, [3], axis=1)
        mat = np.asmatrix(mat.round(15))
        return mat
    else:
        raise ValueError('Value must be a rotation matrix ')


# ---------------------------------------------------------------------------------------#
def rot2(theta, unit='rad'):
    """
    ROT2 SO(2) Rotational Matrix

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :return: rotational matrix (2x2)

    ROT2(THETA) is an SO(2) rotation matrix (2x2) representing a rotation of THETA radians.
    ROT2(THETA, 'deg') as above but THETA is in degrees.
    """
    check_args.unit_check(unit)
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[ct, -st], [st, ct]])
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def trot2(theta, unit='rad'):
    """
    TROT2 SE2 rotation matrix

    :param theta: rotation in radians or degrees
    :param unit: "rad" or "deg" to indicate unit being used
    :return: homogeneous transform matrix (3x3)

    TROT2(THETA) is a homogeneous transformation (3x3) representing a rotation of
    THETA radians.
    TROT2(THETA, 'deg') as above but THETA is in degrees.
    Notes::
    - Translational component is zero.
    """
    tm = rot2(theta, unit)
    tm = np.r_[tm, np.zeros((1, 2))]
    mat = np.c_[tm, np.array([[0], [0], [1]])]
    return mat


# ---------------------------------------------------------------------------------------#
def rpy2r(thetas, order='zyx', unit='rad'):
    """
    RPY2R Roll-pitch-yaw angles to rotation matrix

    :param thetas: list of angles
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
    check_args.unit_check(unit)
    check_args.rpy2r(theta=thetas, order=order)
    if type(thetas[0]) is float or type(thetas[0]) is int:
        # TODO
        # enforce if one element is list.
        # All are list. OR one element is int or float then all are either int or float
        thetas = [thetas]  # Put list in a list

    if unit == 'deg':
        thetas = [[(angles * math.pi / 180) for angles in each_rpy] for each_rpy in thetas]
    if type(thetas[0]) is list:
        roll = [theta[0] for theta in thetas]
        pitch = [theta[1] for theta in thetas]
        yaw = [theta[2] for theta in thetas]

        if order == 'xyz' or order == 'arm':
            x = [rotx(theta) for theta in yaw]
            y = [roty(theta) for theta in pitch]
            z = [rotz(theta) for theta in roll]
            xyz = [(x[i] * y[i] * z[i]) for i in range(len(thetas))]
            xyz = [np.asmatrix(each.round(15)) for each in xyz]
            if len(xyz) == 1:
                return xyz[0]
            else:
                return xyz
        if order == 'zyx' or order == 'vehicle':
            z = [rotz(theta) for theta in yaw]
            y = [roty(theta) for theta in pitch]
            x = [rotx(theta) for theta in roll]
            zyx = [(z[i] * y[i] * x[i]) for i in range(len(thetas))]
            zyx = [np.asmatrix(each.round(15)) for each in zyx]
            if len(zyx) == 1:
                return zyx[0]
            else:
                return zyx
        if order == 'yxz' or order == 'camera':
            y = [roty(theta) for theta in yaw]
            x = [rotx(theta) for theta in pitch]
            z = [rotz(theta) for theta in roll]
            yxz = [(y[i] * x[i] * z[i]) for i in range(len(thetas))]
            yxz = [np.asmatrix(each.round(15)) for each in yxz]
            if len(yxz) == 1:
                return yxz[0]
            else:
                return yxz
    else:
        raise TypeError('thetas must be a list of roll pitch yaw angles\n'
                        'OR a list of list of roll pitch yaw angles.')


# ---------------------------------------------------------------------------------------#
def rpy2tr(thetas, order='zyx', unit='rad'):
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
    rot = rpy2r(thetas, order, unit)
    if type(rot) is list:
        rot = [r2t(each) for each in rot]
    else:
        rot = r2t(rot)
    return rot


# ---------------------------------------------------------------------------------------#
def skew(v):
    """
    SKEW creates Skew-symmetric metrix from vector

    :param v: 1 or 3 vector
    :return: skew-symmetric matrix
    SKEW(V) is a skew-symmetric matrix formed from V.

    If V (1x1) then S =
            | 0  -v |
            | v   0 |

    and if V (1x3) then S =
           |  0  -vz   vy |
           | vz    0  -vx |
           |-vy   vx    0 |

    Notes::
    - This is the inverse of the function VEX().
    - These are the generator matrices for the Lie algebras so(2) and so(3).
    """
    if common.isvec(v, 3):
        s = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
    elif common.isvec(v, 1):
        s = np.matrix([[0, -v[0, 0]], [v[0, 0], 0]])
    else:
        raise AttributeError("argument must be a 1- or 3-vector")
    return s


# ---------------------------------------------------------------------------------------#
def skewa(s):
    """
    SKEWA creates augmented skew-symmetric matrix

    :param s: 3 or 6 vector
    :return: augmented skew-symmetric matrix

    SKEWA(V) is an augmented skew-symmetric matrix formed from V.

    If V (1x3) then S =
           |  0  -v3  v1 |
           | v3    0  v2 |
           |  0    0   0 |

    and if V (1x6) then S =
           |  0  -v6   v5  v1 |
           | v6    0  -v4  v2 |
           |-v5   v4    0  v3 |
           |  0    0    0   0 |

    Notes::
    - This is the inverse of the function VEXA().
    - These are the generator matrices for the Lie algebras se(2) and se(3).
    - Map twist vectors in 2D and 3D space to se(2) and se(3).
    """
    s = s.flatten(1)
    if s.size == 3:
        omega = np.concatenate((skew(np.matrix(s[0, 2])), [[s[0, 0]], [s[0, 1]]]), axis=1)
        omega = np.concatenate((omega, [[0, 0, 0]]), axis=0)
        return omega
    elif s.size == 6:
        omega = np.concatenate((skew(np.matrix([s[0, 3], s[0, 4], s[0, 5]])), [[s[0, 0]], [s[0, 1]], [s[0, 2]]]),
                               axis=1)
        omega = np.concatenate((omega, [[0, 0, 0, 0]]), axis=0)
        return omega
    else:
        raise AttributeError("expecting a 3- or 6-vector")


# ---------------------------------------------------------------------------------------#
def unitize(v):
    """
    UNIT Unitize a vector

    :param v: given unit vector
    :return: a unit-vector parallel to V.

    Reports error for the case where V is non-symbolic and norm(V) is zero
    """
    n = np.linalg.norm(v, "fro")
    # Todo ISA
    if n > np.spacing([1])[0]:
        return v / n
    else:
        raise AttributeError("Vector has zero norm")


# ---------------------------------------------------------------------------------------#
def angvec2r(theta, v):
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
    if np.isscalar(theta) is False or common.isvec(v) is False:
        raise AttributeError("Arguments must be theta and vector")
    # TODO implement ISA
    elif np.linalg.norm(v) < 10 * np.spacing([1])[0]:
        if False:
            raise AttributeError("Bad arguments")
        else:
            return np.eye(3)
    sk = skew(np.matrix(unitize(v)))
    m = np.eye(3) + np.sin(theta) * sk + (1 - np.cos(theta)) * sk * sk
    return m


# ---------------------------------------------------------------------------------------#
def angvec2tr(theta, v):
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
    return r2t(angvec2r(theta, v))


# ---------------------------------------------------------------------------------------#
def vex(s):
    """
    VEX Convert skew-symmetric matrix to vector

    :param s:skew-symmetric matrix
    :return: vector

    VEX(S) is the vector which has the corresponding skew-symmetric matrix S.
    In the case that S (2x2) then V is 1x1
           S = | 0  -v |
               | v   0 |
    In the case that S (3x3) then V is 3x1.
               |  0  -vz   vy |
           S = | vz    0  -vx |
               |-vy   vx    0 |

    Notes::
    - This is the inverse of the function SKEW().
    - Only rudimentary checking (zero diagonal) is done to ensure that the matrix
    is actually skew-symmetric.
    - The function takes the mean of the two elements that correspond to each unique
    element of the matrix.
    """
    if s.shape == (3, 3):
        return 0.5 * np.matrix([[s[2, 1] - s[1, 2]], [s[0, 2] - s[2, 0]], [s[1, 0] - s[0, 1]]])
    elif s.shape == (2, 2):
        return 0.5 * np.matrix([[s[1, 0] - s[0, 1]]])
    else:
        raise AttributeError("Argument must be 2x2 or 3x3 matrix")


# ---------------------------------------------------------------------------------------#
def tr2rt(t):
    """
    TR2RT Convert homogeneous transform to rotation and translation

    :param t: homogeneous transform matrix
    :return: Rotation and translation of the homogeneous transform

    TR2RT(TR) splits a homogeneous transformation matrix (NxN) into an orthonormal
    rotation matrix R (MxM) and a translation vector T (Mx1), where N=M+1.

    Works for TR in SE(2) or SE(3)
    - If TR is 4x4, then R is 3x3 and T is 3x1.
    - If TR is 3x3, then R is 2x2 and T is 2x1.
    """
    if t.shape == (4, 4):
        assert t.shape[0] == t.shape[1]
        R = t2r(t)
        t = np.matrix([[t[0, 3]], [t[1, 3]], [t[2, 3]]])
        return [R, t]
    else:
        assert t.shape[0] == t.shape[1]
        R = t2r(t)
        t = np.matrix([[t[0, 2]], [t[1, 2]]])
        return [R, t]


# ---------------------------------------------------------------------------------------#
def rt2tr(r, t):
    """
    RT2TR Convert rotation and translation to homogeneous transform

    :param r: rotation matrix
    :param t: translation
    :return: homogeneous transform

    RT2TR(R, t) is a homogeneous transformation matrix (N+1xN+1) formed from an
    orthonormal rotation matrix R (NxN) and a translation vector t
    (Nx1).  Works for R in SO(2) or SO(3):
    - If R is 2x2 and t is 2x1, then TR is 3x3
    - If R is 3x3 and t is 3x1, then TR is 4x4
    """
    if r.shape == (2, 2):
        if r.shape[0] != r.shape[1]:
            raise AttributeError("R must be square")
        if r.shape[0] != t.shape[0]:
            raise AttributeError("R and t must have the same number of rows")
        tr = np.concatenate((r, t), axis=1)
        tr = np.concatenate((tr, np.matrix([[0, 0, 1]])), axis=0)
        return tr
    else:
        if r.shape[0] != r.shape[1]:
            raise AttributeError("R must be square")
        if r.shape[0] != t.shape[0]:
            raise AttributeError("R and t must have the same number of rows")
        tr = np.concatenate((r, t), axis=1)
        tr = np.concatenate((tr, np.matrix([[0, 0, 0, 1]])), axis=0)
        return tr


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


# ------------------------------------------------------------------------------------------------------------------- #
def tr2angvec(tr, unit='rad'):
    """
    TR2ANGVEC Convert rotation matrix to angle-vector form
    :param tr: Rotation matrix
    :param unit: 'rad' or 'deg'
    :return: Angle-vector form
    TR2ANGVEC(R, OPTIONS) is rotation expressed in terms of an angle THETA (1x1) about the axis V (1x3) equivalent to the orthonormal rotation matrix R (3x3).
    TR2ANGVEC(T, OPTIONS) as above but uses the rotational part of the homogeneous transform T (4x4).
    If R (3x3xK) or T (4x4xK) represent a sequence then THETA (Kx1)is a vector of angles for corresponding elements of the sequence and V (Kx3) are the corresponding axes, one per row.
    Options::
    'deg'   Return angle in degrees
    Notes::
    - For an identity rotation matrix both THETA and V are set to zero.
    - The rotation angle is always in the interval [0 pi], negative rotation is handled by inverting the direction of the rotation axis.
    - If no output arguments are specified the result is displayed.
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
def tr2eul(tr, unit='rad', flip=False):
    """
    TR2EUL Convert homogeneous transform to Euler angles
    :param tr: Homogeneous transformation
    :param unit: 'rad' or 'deg'
    :param flip: True or False
    :return: Euler angles
    TR2EUL(T, OPTIONS) are the ZYZ Euler angles (1x3) corresponding to the rotational part of a homogeneous transform T (4x4). The 3 angles EUL=[PHI,THETA,PSI] correspond to sequential rotations about the Z, Y and Z axes respectively.
    TR2EUL(R, OPTIONS) as above but the input is an orthonormal rotation matrix R (3x3).
    If R (3x3xK) or T (4x4xK) represent a sequence then each row of EUL corresponds to a step of the sequence.
    Options::
    'deg'   Compute angles in degrees (radians default)
    'flip'  Choose first Euler angle to be in quadrant 2 or 3.
    Notes::
    - There is a singularity for the case where THETA=0 in which case PHI is arbitrarily set to zero and PSI is the sum (PHI+PSI).
    - Translation component is ignored.
    """
    check_args.unit_check(unit)
    check_args.tr2eul(tr=tr, unit=unit, flip=flip)

    if tr.ndim > 2:
        eul = np.zeros([tr.shape[2], 3])
        for i in range(0, tr.shape[2]):
            eul[i, :] = tr2eul(tr[i, :, :])
        return eul
    else:
        eul = np.zeros([1, 3])

    if abs(tr[0, 2]) < np.spacing([1])[0] and abs(tr[1, 2]) < np.spacing([1])[0]:
        eul[0, 0] = 0
        sp = 0
        cp = 0
        eul[0, 1] = math.atan2(cp * tr[0, 2] + sp * tr[1, 2], tr[2, 2])
        eul[0, 2] = math.atan2(-sp * tr[0, 0] + cp * tr[1, 0], -sp * tr[0, 1] + cp * tr[1, 1])
    else:
        if flip:
            eul[0, 0] = math.atan2(-tr[1, 2], -tr[0, 2])
        else:
            eul[0, 0] = math.atan2(tr[1, 2], tr[0, 2])
        sp = math.sin(eul[0, 0])
        cp = math.cos(eul[0, 0])
        eul[0, 1] = math.atan2(cp * tr[0, 2] + sp * tr[1, 2], tr[2, 2])
        eul[0, 2] = math.atan2(-sp * tr[0, 0] + cp * tr[1, 0], -sp * tr[0, 1] + cp * tr[1, 1])

    if unit == 'deg':
        eul = eul * 180 / math.pi

    return eul


# ------------------------------------------------------------------------------------------------------------------- #
def tr2rpy(tr, unit='rad', order='zyx'):
    """
    TR2RPY Convert a homogeneous transform to roll-pitch-yaw angles
    :param tr: Homogeneous transformation
    :param unit: 'rad' or 'deg'
    :param order: 'xyz', 'zyx' or 'yxz'
    :return: Roll-pitch-yaw angle
    TR2RPY(T, options) are the roll-pitch-yaw angles (1x3) corresponding to the rotation part of a homogeneous transform T. The 3 angles RPY=[R,P,Y] correspond to sequential rotations about the Z, Y and X axes respectively.
    TR2RPY(R, options) as above but the input is an orthonormal rotation matrix R (3x3).
    If R (3x3xK) or T (4x4xK) represent a sequence then each row of RPY corresponds to a step of the sequence.
    Options::
    'deg'   Compute angles in degrees (radians default)
    'xyz'   Return solution for sequential rotations about X, Y, Z axes
    'yxz'   Return solution for sequential rotations about Y, X, Z axes
    Notes::
    - There is a singularity for the case where P=pi/2 in which case R is arbitrarily set to zero and Y is the sum (R+Y).
    - Translation component is ignored.
    - Toolbox rel 8-9 has the reverse default angle sequence as default
    """
    check_args.unit_check(unit)
    check_args.tr2rpy(tr=tr, unit=unit, order=order)

    if tr.ndim > 2:
        rpy = np.zeros([tr.shape[2], 3])
        for i in range(0, tr.shape[2]):
            rpy[i, :] = tr2rpy(tr[i, :, :])
        return rpy
    else:
        rpy = np.zeros([1, 3])

    if common.isrot(tr) or common.ishomog(tr, dim=[4, 4]):
        if order == 'xyz' or order == 'arm':
            if abs(abs(tr[0, 2]) - 1) < np.spacing([1])[0]:
                rpy[0, 0] = 0
                rpy[0, 1] = math.asin(tr[0, 2])
                if tr[0, 2] > 0:
                    rpy[0, 2] = math.atan2(tr[2, 1], tr[1, 1])
                else:
                    rpy[0, 2] = -math.atan2(tr[1, 0], tr[2, 0])
            else:
                rpy[0, 0] = -math.atan2(tr[0, 1], tr[0, 0])
                rpy[0, 1] = math.atan2(tr[0, 2] * math.cos(rpy[0, 0]), tr[0, 0])
                rpy[0, 2] = -math.atan2(tr[1, 2], tr[2, 2])
        if order == 'zyx' or order == 'vehicle':
            if abs(abs(tr[2, 0]) - 1) < np.spacing([1])[0]:
                rpy[0, 0] = 0
                rpy[0, 1] = -math.asin(tr[2, 0])
                if tr[2, 0] < 0:
                    rpy[0, 2] = -math.atan2(tr[0, 1], tr[0, 2])
                else:
                    rpy[0, 2] = math.atan2(-tr[0, 1], -tr[0, 2])
            else:
                rpy[0, 0] = math.atan2(tr[2, 1], tr[2, 2])
                rpy[0, 1] = math.atan2(-tr[2, 0] * math.cos(rpy[0, 0]), tr[2, 2])
                rpy[0, 2] = math.atan2(tr[1, 0], tr[0, 0])
        if order == 'yxz' or order == 'camera':
            if abs(abs(tr[1, 2]) - 1) < np.spacing([1])[0]:
                rpy[0, 0] = 0
                rpy[0, 1] = -math.asin(tr[1, 2])
                if tr[1, 2] < 0:
                    rpy[0, 2] = -math.atan2(tr[2, 0], tr[0, 0])
                else:
                    rpy[0, 2] = math.atan2(-tr[2, 0], -tr[2, 1])
            else:
                rpy[0, 0] = math.atan2(tr[1, 0], tr[1, 1])
                rpy[0, 1] = math.atan2(-math.cos(rpy[0, 0]) * tr[1, 2], tr[1, 1])
                rpy[0, 2] = math.atan2(tr[0, 2], tr[2, 2])
    else:
        raise TypeError('Argument must be a 3x3 or 4x4 matrix.')

    if unit == 'deg':
        rpy = rpy * 180 / math.pi

    return rpy


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
    if common.ishomog(S, [4, 4]) or common.isvec(S, 6):
        if theta is None:
            if common.isvec(S, 6):
                S = skewa(S)
            return expm(S)
        else:
            if S.shape == (4, 4):
                [skw, v] = tr2rt(S)
            else:
                v = S[:, 0]
                skw = skew(np.matrix([[S[0, 3], S[0, 4], S[0, 5]]]))
            R = trexp(skw, theta)
            t = (np.eye(3) + np.sin(theta) * skw + (1 - np.cos(theta)) * skw * skw) * v
            return rt2tr(R, t)
    elif common.ishomog(S, [3, 3]) or common.isvec(S, 3):
        if common.isrot(S):
            w = vex(S)
        elif common.isvec(S):
            w = S
        else:
            raise AttributeError("Bad arguments, expectin 1x3 or 3x3")
        if np.linalg.norm(w) < 10 * np.spacing([1])[0]:
            return np.eye(3)
        if theta is None:
            theta = np.linalg.norm(w)
            w = unitize(w)
        else:
            i = 0
            # todo ISUNIT
        S = skew(w)
        T = np.eye(3) + np.sin(theta) * S + (1 - np.cos(theta)) * S ** 2
        return T
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


# ---------------------------------------------------------------------------------------#
def oa2r(o, a=None):
    """
    OA2R Convert orientation and approach vectors to rotation matrix

    :param o: vector parallel to Y- axes
    :param a: vector parallel to the z-axes
    :return: rotation matrix

    R = OA2R(O, A) is an SO(3) rotation matrix (3x3) for the specified orientation
    and approach vectors (3x1) formed from 3 vectors such that R = [N O A] and N = O x A.

    Notes::
    - The matrix is guaranteed to be orthonormal so long as O and A are not parallel.
    - The vectors O and A are parallel to the Y- and Z-axes of the coordinate frame.
    """
    n = np.cross(o, a)
    o = np.cross(a, n)
    i = unitize(n)
    j = unitize(o)
    k = unitize(a)
    R = np.matrix([[i[0, 0], j[0, 0], k[0, 0]], [i[0, 1], j[0, 1], k[0, 1]], [i[0, 2], j[0, 2], k[0, 2]]])
    return R


# ---------------------------------------------------------------------------------------#
def oa2tr(o, a=None):
    """
    OA2TR Convert orientation and approach vectors to homogeneous transformation

    :param o: vector parallel to Y- axes
    :param a: vector parallel to the z-axes
    :return: homogeneous transform

    T = OA2TR(O, A) is an SE(3) homogeneous tranformation (4x4) for the
    specified orientation and approach vectors (3x1) formed from 3 vectors
    such that R = [N O A] and N = O x A.

    Notes::
    - The rotation submatrix is guaranteed to be orthonormal so long as O and A
    are not parallel.
    - The translational part is zero.
    - The vectors O and A are parallel to the Y- and Z-axes of the coordinate frame.
    """
    return r2t(oa2r(o, a))


# ---------------------------------------------------------------------------------------#
def transl(x=None, y=None, z=None):
    """
    TRANSL Create or unpack an SE(3) translational homogeneous transform

    :param x: translation along x axes, homogeneous transform or a list of translations
    :param y: translation along y axes
    :param z: translation along z axes
    :return: homogeneous transform with pure translation

    Create a translational SE(3) matrix::

    T = TRANSL(X, Y, Z) is an SE(3) homogeneous transform (4x4) representing a
    pure translation of X, Y and Z.

    T = TRANSL(P) is an SE(3) homogeneous transform (4x4) representing a
    translation of P=[X,Y,Z]. If P (Mx3) it represents a sequence and T (4x4xM)
    is a sequence of homogeneous transforms such that T(:,:,i) corresponds to
    the i'th row of P.

    Extract the translational part of an SE(3) matrix::

    P = TRANSL(T) is the translational part of a homogeneous transform T as a
    3-element column vector.  If T (4x4xM) is a homogeneous transform sequence
    the rows of P (Mx3) are the translational component of the corresponding
    transform in the sequence.

    [X,Y,Z] = TRANSL(T) is the translational part of a homogeneous transform
    T as three components.  If T (4x4xM) is a homogeneous transform sequence
    then X,Y,Z (1xM) are the translational components of the corresponding
    transform in the sequence.
   """
    if type(x) is np.matrix:
        if common.ishomog(x, [4, 4]):
            return x[:3, 2]
    elif type(x) is list:
        if len(x) == 3:
            temp = np.matrix([[x[0]], [x[1]], [x[2]]])
            temp = np.concatenate((np.eye(3), temp), axis=1)
            return np.concatenate((temp, np.matrix([[0, 0, 0, 1]])), axis=0)
    # todo trajectory case
    elif x is not None and y is not None and z is not None:
        t = np.matrix([[x], [y], [z]])
        return rt2tr(np.eye(3), t)
    else:
        raise AttributeError("Invalid arguments")


# ---------------------------------------------------------------------------------------#
def transl2(x=None, y=None):
    """
    TRANSL2 Create or unpack an SE(2) translational homogeneous transform

    :param x: x translation, homogeneous transform or a list of translations
    :param y: y translation
    :return: homogeneous transform matrix or the translation elements of a
    homogeneous transform

    Create a translational SE(2) matrix::

    T = TRANSL2(X, Y) is an SE(2) homogeneous transform (3x3) representing a
    pure translation.

    T = TRANSL2(P) is a homogeneous transform representing a translation or
    point P=[X,Y]. If P (Mx2) it represents a sequence and T (3x3xM) is a
    sequence of homogeneous transforms such that T(:,:,i) corresponds to the
    i'th row of P.

    Extract the translational part of an SE(2) matrix::

    P = TRANSL2(T) is the translational part of a homogeneous transform as a
    2-element column vector.  If T (3x3xM) is a homogeneous transform
    sequence the rows of P (Mx2) are the translational component of the
    corresponding transform in the sequence.
    """
    if type(x) is np.matrix:
        if common.ishomog(x, [3, 3]):
            return x[:2, 2]
    elif type(x) is list:
        if len(x) == 2:
            temp = np.matrix([[x[0]], [x[1]]])
            temp = np.concatenate((np.eye(2), temp), axis=1)
            return np.concatenate((temp, np.matrix([[0, 0, 1]])), axis=0)
    elif x is not None and y is not None:
        t = np.matrix([[x], [y]])
        return rt2tr(np.eye(2), t)


# ---------------------------------------------------------------------------------------#
def eul2r(phi, theta=None, psi=None, unit='rad'):
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
    check_args.unit_check(unit)
    if type(phi) is np.matrix and theta is None and psi is None:
        theta = phi[:, 1]
        psi = phi[:, 2]
        phi = phi[:, 0]
    if theta is None or psi is None:
        raise AttributeError("Invalid arguments, expecting, 3 inputs or 3-vector")

    if unit == 'deg':
        d2r = np.pi / 180
        phi = phi * d2r
        theta = theta * d2r
        psi = psi * d2r

    if phi.shape[0] == 1:
        return rotz(phi[0, 0]) * roty(theta[0, 0]) * rotz(psi[0, 0])
    else:
        R = [phi.shape[0]]
        for i in range(0, phi.shape[0]):
            R[i] = (rotz(phi[i, 0]) * roty(theta[i, 0]) * rotz(psi[i, 0]))
        return R


# ---------------------------------------------------------------------------------------#
def eul2tr(phi, theta=None, psi=None, unit='rad'):
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
    R = eul2r(phi, theta, psi, unit)
    return r2t(R)


# ---------------------------------------------------------------------------------------#
def np2vtk(mat):
    if mat.shape == (4, 4):
        obj = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                obj.SetElement(i, j, mat[i, j])
        return obj


# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':
    # When run as main, initialise test cases in test_classes_to_tun and runs them
    # Refer
    # https://stackoverflow.com/questions/5360833/how-to-run-multiple-classes-in-single-test-suite-in-python-unit-testing
    test_classes_to_run = [test_transforms.TestRotx]

    loader = unittest.TestLoader()

    suits_list = []
    for test_class in test_classes_to_run:
        suits_list.append(loader.loadTestsFromTestCase(test_class))

    big_suite = unittest.TestSuite(suits_list)

    runner = unittest.TextTestRunner(verbosity=2)
    results = runner.run(big_suite)
