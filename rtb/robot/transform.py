"""
Primitive operations for 3x3 orthonormal and 4x4 homogeneous matrices.

Python implementation by: Luis Fernando Lara Tobar and Peter Corke.
Based on original Robotics Toolbox for Matlab code by Peter Corke.
Permission to use and copy is granted provided that acknowledgement of
the authors is made.

@author: Luis Fernando Lara Tobar and Peter Corke
"""

from numpy import *
from robot.utility import *
from numpy.linalg import norm
import robot.Quaternion as Q

def rotx(theta):
    """
    Rotation about X-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about X-axis

    @see: L{roty}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)
    return mat([[1,  0,    0],
            [0,  ct, -st],
            [0,  st,  ct]])

def roty(theta):
    """
    Rotation about Y-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Y-axis

    @see: L{rotx}, L{rotz}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)

    return mat([[ct,   0,   st],
            [0,    1,    0],
            [-st,  0,   ct]])

def rotz(theta):
    """
    Rotation about Z-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Z-axis

    @see: L{rotx}, L{roty}, L{rotvec}
    """
    
    ct = cos(theta)
    st = sin(theta)

    return mat([[ct,      -st,  0],
            [st,       ct,  0],
            [ 0,    0,  1]])

def trotx(theta):
    """
    Rotation about X-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation about X-axis

    @see: L{troty}, L{trotz}, L{rotx}
    """
    return r2t(rotx(theta))

def troty(theta):
    """
    Rotation about Y-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation about Y-axis

    @see: L{troty}, L{trotz}, L{roty}
    """
    return r2t(roty(theta))

def trotz(theta):
    """
    Rotation about Z-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation about Z-axis

    @see: L{trotx}, L{troty}, L{rotz}
    """
    return r2t(rotz(theta))


##################### Euler angles

def tr2eul(m):
    """
    Extract Euler angles.
    Returns a vector of Euler angles corresponding to the rotational part of 
    the homogeneous transform.  The 3 angles correspond to rotations about
    the Z, Y and Z axes respectively.
    
    @type m: 3x3 or 4x4 matrix
    @param m: the rotation matrix
    @rtype: 1x3 matrix
    @return: Euler angles [S{theta} S{phi} S{psi}]
    
    @see:  L{eul2tr}, L{tr2rpy}
    """
    
    try:
        m = mat(m)
        if ishomog(m):
            euler = mat(zeros((1,3)))
            if norm(m[0,2])<finfo(float).eps and norm(m[1,2])<finfo(float).eps:
                # singularity
                euler[0,0] = 0
                sp = 0
                cp = 1
                euler[0,1] = arctan2(cp*m[0,2] + sp*m[1,2], m[2,2])
                euler[0,2] = arctan2(-sp*m[0,0] + cp*m[1,0], -sp*m[0,1] + cp*m[1,1])
                return euler
            else:
                euler[0,0] = arctan2(m[1,2],m[0,2])
                sp = sin(euler[0,0])
                cp = cos(euler[0,0])
                euler[0,1] = arctan2(cp*m[0,2] + sp*m[1,2], m[2,2])
                euler[0,2] = arctan2(-sp*m[0,0] + cp*m[1,0], -sp*m[0,1] + cp*m[1,1])
                return euler
            
    except ValueError:
        euler = []
        for i in range(0,len(m)):
            euler.append(tr2eul(m[i]))
        return euler
        

def eul2r(phi, theta=None, psi=None):
    """
    Rotation from Euler angles.
    
    Two call forms:
        - R = eul2r(S{theta}, S{phi}, S{psi})
        - R = eul2r([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, Z axes respectively.

    @type phi: number or list/array/matrix of angles
    @param phi: the first Euler angle, or a list/array/matrix of angles
    @type theta: number
    @param theta: the second Euler angle
    @type psi: number
    @param psi: the third Euler angle
    @rtype: 3x3 orthonormal matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2eul}, L{eul2tr}, L{tr2rpy}

    """

    n = 1
    if theta == None and psi==None:
        # list/array/matrix argument
        phi = mat(phi)
        if numcols(phi) != 3:
            error('bad arguments')
        else:
            n = numrows(phi)
            psi = phi[:,2]
            theta = phi[:,1]
            phi = phi[:,0]
    elif (theta!=None and psi==None) or (theta==None and psi!=None):
        error('bad arguments')
    elif not isinstance(phi,(int,int32,float,float64)):
        # all args are vectors
        phi = mat(phi)
        n = numrows(phi)
        theta = mat(theta)
        psi = mat(psi)

    if n>1:
        R = []
        for i in range(0,n):
                r = rotz(phi[i,0]) * roty(theta[i,0]) * rotz(psi[i,0])
                R.append(r)
        return R
    try:
        r = rotz(phi[0,0]) * roty(theta[0,0]) * rotz(psi[0,0])
        return r
    except:
        r = rotz(phi) * roty(theta) * rotz(psi)
        return r

def eul2tr(phi,theta=None,psi=None):
    """
    Rotation from Euler angles.
    
    Two call forms:
        - R = eul2tr(S{theta}, S{phi}, S{psi})
        - R = eul2tr([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, Z axes respectively.

    @type phi: number or list/array/matrix of angles
    @param phi: the first Euler angle, or a list/array/matrix of angles
    @type theta: number
    @param theta: the second Euler angle
    @type psi: number
    @param psi: the third Euler angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2eul}, L{eul2r}, L{tr2rpy}

    """
    return r2t( eul2r(phi, theta, psi) )


################################## RPY angles


def tr2rpy(m):
    """
    Extract RPY angles.
    Returns a vector of RPY angles corresponding to the rotational part of 
    the homogeneous transform.  The 3 angles correspond to rotations about
    the Z, Y and X axes respectively.
    
    @type m: 3x3 or 4x4 matrix
    @param m: the rotation matrix
    @rtype: 1x3 matrix
    @return: RPY angles [S{theta} S{phi} S{psi}]
    
    @see:  L{rpy2tr}, L{tr2eul}
    """
    try:
        m = mat(m)
        if ishomog(m):
            rpy = mat(zeros((1,3)))
            if norm(m[0,0])<finfo(float).eps and norm(m[1,0])<finfo(float).eps:
                # singularity
                rpy[0,0] = 0
                rpy[0,1] = arctan2(-m[2,0], m[0,0])
                rpy[0,2] = arctan2(-m[1,2], m[1,1])
                return rpy
            else:
                rpy[0,0] = arctan2(m[1,0],m[0,0])
                sp = sin(rpy[0,0])
                cp = cos(rpy[0,0])
                rpy[0,1] = arctan2(-m[2,0], cp*m[0,0] + sp*m[1,0])
                rpy[0,2] = arctan2(sp*m[0,2] - cp*m[1,2], cp*m[1,1] - sp*m[0,1])
                return rpy
            
    except ValueError:
        rpy = []
        for i in range(0,len(m)):
            rpy.append(tr2rpy(m[i]))
        return rpy
        
def rpy2r(roll, pitch=None,yaw=None):
    """
    Rotation from RPY angles.
    
    Two call forms:
        - R = rpy2r(S{theta}, S{phi}, S{psi})
        - R = rpy2r([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, X axes respectively.

    @type roll: number or list/array/matrix of angles
    @param roll: roll angle, or a list/array/matrix of angles
    @type pitch: number
    @param pitch: pitch angle
    @type yaw: number
    @param yaw: yaw angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2rpy}, L{rpy2r}, L{tr2eul}

    """
    n=1
    if pitch==None and yaw==None:
        roll= mat(roll)
        if numcols(roll) != 3:
            error('bad arguments')
        n = numrows(roll)
        pitch = roll[:,1]
        yaw = roll[:,2]
        roll = roll[:,0]
    if n>1:
        R = []
        for i in range(0,n):
            r = rotz(roll[i,0]) * roty(pitch[i,0]) * rotx(yaw[i,0])
            R.append(r)
        return R
    try:
        r = rotz(roll[0,0]) * roty(pitch[0,0]) * rotx(yaw[0,0])
        return r
    except:
        r = rotz(roll) * roty(pitch) * rotx(yaw)
        return r


def rpy2tr(roll, pitch=None, yaw=None):
    """
    Rotation from RPY angles.
    
    Two call forms:
        - R = rpy2tr(r, p, y)
        - R = rpy2tr([r, p, y])
    These correspond to rotations about the Z, Y, X axes respectively.

    @type roll: number or list/array/matrix of angles
    @param roll: roll angle, or a list/array/matrix of angles
    @type pitch: number
    @param pitch: pitch angle
    @type yaw: number
    @param yaw: yaw angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2rpy}, L{rpy2r}, L{tr2eul}

    """
    return r2t( rpy2r(roll, pitch, yaw) )

###################################### OA vector form


def oa2r(o,a):
    """Rotation from 2 vectors.
    The matrix is formed from 3 vectors such that::
        R = [N O A] and N = O x A.  

    In robotics A is the approach vector, along the direction of the robot's 
    gripper, and O is the orientation vector in the direction between the 
    fingertips.
    
    The submatrix is guaranteed to be orthonormal so long as O and A are 
    not parallel.
    
    @type o: 3-vector
    @param o: The orientation vector.
    @type a: 3-vector
    @param a: The approach vector
    @rtype: 3x3 orthonormal rotation matrix
    @return: Rotatation matrix
    
    @see: L{rpy2r}, L{eul2r}
    """
    n = crossp(o, a)
    n = unit(n)
    o = crossp(a, n);
    o = unit(o).reshape(3,1)
    a = unit(a).reshape(3,1)
    return bmat('n o a')


def oa2tr(o,a):
    """otation from 2 vectors.
    The rotation submatrix is formed from 3 vectors such that::

        R = [N O A] and N = O x A.  

    In robotics A is the approach vector, along the direction of the robot's 
    gripper, and O is the orientation vector in the direction between the 
    fingertips.
    
    The submatrix is guaranteed to be orthonormal so long as O and A are 
    not parallel.
    
    @type o: 3-vector
    @param o: The orientation vector.
    @type a: 3-vector
    @param a: The approach vector
    @rtype: 4x4 homogeneous transformation matrix
    @return: Transformation matrix
    
    @see: L{rpy2tr}, L{eul2tr}
    """
    return r2t(oa2r(o,a))
    
    
###################################### angle/vector form


def rotvec2r(theta, v):
    """
    Rotation about arbitrary axis.  Compute a rotation matrix representing
    a rotation of C{theta} about the vector C{v}.
    
    @type v: 3-vector
    @param v: rotation vector
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation

    @see: L{rotx}, L{roty}, L{rotz}
    """
    v = arg2array(v);
    ct = cos(theta)
    st = sin(theta)
    vt = 1-ct
    r = mat([[ct,         -v[2]*st,    v[1]*st],\
             [v[2]*st,          ct,   -v[0]*st],\
             [-v[1]*st,  v[0]*st,           ct]])
    return v*v.T*vt+r

def rotvec2tr(theta, v):
    """
    Rotation about arbitrary axis.  Compute a rotation matrix representing
    a rotation of C{theta} about the vector C{v}.
    
    @type v: 3-vector
    @param v: rotation vector
    @type theta: number
    @param theta: the rotation angle
    @rtype: 4x4 homogeneous matrix
    @return: rotation

    @see: L{trotx}, L{troty}, L{trotz}
    """
    return r2t(rotvec2r(theta, v))


###################################### translational transform


def transl(x, y=None, z=None):
    """
    Create or decompose translational homogeneous transformations.
    
    Create a homogeneous transformation
    ===================================
    
        - T = transl(v)
        - T = transl(vx, vy, vz)
        
        The transformation is created with a unit rotation submatrix.
        The translational elements are set from elements of v which is
        a list, array or matrix, or from separate passed elements.
    
    Decompose a homogeneous transformation
    ======================================
    

        - v = transl(T)   
    
        Return the translation vector
    """
           
    if y==None and z==None:
            x=mat(x)
            try:
                    if ishomog(x):
                            return x[0:3,3].reshape(3,1)
                    else:
                            return concatenate((concatenate((eye(3),x.reshape(3,1)),1),mat([0,0,0,1])))
            except AttributeError:
                    n=len(x)
                    r = [[],[],[]]
                    for i in range(n):
                            r = concatenate((r,x[i][0:3,3]),1)
                    return r
    elif y!=None and z!=None:
            return concatenate((concatenate((eye(3),mat([x,y,z]).T),1),mat([0,0,0,1])))


###################################### Skew symmetric transform


def skew(*args):
    """
    Convert to/from skew-symmetric form.  A skew symmetric matrix is a matrix
    such that M = -M'
    
    Two call forms
    
        -ss = skew(v)
        -v = skew(ss)
        
    The first form builds a 3x3 skew-symmetric from a 3-element vector v.
    The second form takes a 3x3 skew-symmetric matrix and returns the 3 unique
    elements that it contains.
    
    """
    
    def ss(b):
        return  matrix([
            [0, -b[2],  b[1]],
            [b[2],  0,  -b[0]],
            [-b[1], b[0],   0]]);

    if len(args) == 1:
        # convert matrix to skew vector
        b = args[0];
        
        if isrot(b):
            return 0.5*matrix( [b[2,1]-b[1,2], b[0,2]-b[2,0], b[1,0]-b[0,1]] );
        elif ishomog(b):
            return vstack( (b[0:3,3], 0.5*matrix( [b[2,1]-b[1,2], b[0,2]-b[2,0], b[1,0]-b[0,1]] ).T) );

    
    # build skew-symmetric matrix
          
        b = arg2array(b);
        if len(b) == 3:
            return ss(b);
        elif len(b) == 6:
            r = hstack( (ss(b[3:6]), mat(b[0:3]).T) );
            r = vstack( (r, mat([0, 0, 0, 1])) );
            return r;
            
    elif len(args) == 3:
            return ss(args);    
    elif len(args) == 6:
            r = hstack( (ss(args[3:6]), mat(args[0:3]).T) );
            r = vstack( (r, mat([0, 0, 0, 1])) );
            return r;    
    else:
        raise ValueError;



def tr2diff(t1, t2):
    """
    Convert a transform difference to differential representation.
    Returns the 6-element differential motion required to move
    from T1 to T2 in base coordinates.
    
    @type t1: 4x4 homogeneous transform
    @param t1: Initial value
    @type t2: 4x4 homogeneous transform
    @param t2: Final value
    @rtype: 6-vector
    @return: Differential motion [dx dy dz drx dry drz]
    @see: L{skew}
    """
    
    t1 = mat(t1)
    t2 = mat(t2)
    
    d = concatenate(
        (t2[0:3,3]-t1[0:3,3],
        0.5*(   crossp(t1[0:3,0], t2[0:3,0]) +
                crossp(t1[0:3,1], t2[0:3,1]) +
                crossp(t1[0:3,2], t2[0:3,2]) )
            ))
    return d

################################## Utility


def trinterp(T0, T1, r):
    """
    Interpolate homogeneous transformations.
    Compute a homogeneous transform interpolation between C{T0} and C{T1} as
    C{r} varies from 0 to 1 such that::
    
        trinterp(T0, T1, 0) = T0
        trinterp(T0, T1, 1) = T1
        
    Rotation is interpolated using quaternion spherical linear interpolation.

    @type T0: 4x4 homogeneous transform
    @param T0: Initial value
    @type T1: 4x4 homogeneous transform
    @param T1: Final value
    @type r: number
    @param r: Interpolation index, in the range 0 to 1 inclusive
    @rtype: 4x4 homogeneous transform
    @return: Interpolated value
    @see: L{quaternion}, L{ctraj}
    """
    
    q0 = Q.quaternion(T0)
    q1 = Q.quaternion(T1)
    p0 = transl(T0)
    p1 = transl(T1)

    qr = q0.interp(q1, r)
    pr = p0*(1-r) + r*p1

    return vstack( (concatenate((qr.r(),pr),1), mat([0,0,0,1])) )


def trnorm(t):
    """
    Normalize a homogeneous transformation.
    Finite word length arithmetic can cause transforms to become `unnormalized',
    that is the rotation submatrix is no longer orthonormal (det(R) != 1).
    
    The rotation submatrix is re-orthogonalized such that the approach vector
    (third column) is unchanged in direction::
    
        N = O x A
        O = A x N

    @type t: 4x4 homogeneous transformation
    @param t: the transform matrix to convert
    @rtype: 3x3 orthonormal rotation matrix
    @return: rotation submatrix
    @see: L{oa2tr}
    @bug: Should work for 3x3 matrix as well.
    """

    t = mat(t)      # N O A
    n = crossp(t[0:3,1],t[0:3,2]) # N = O X A
    o = crossp(t[0:3,2],t[0:3,0]) # O = A x N
    return concatenate(( concatenate((unit(n),unit(t[0:3,1]),unit(t[0:3,2]),t[0:3,3]),1),
         mat([0,0,0,1])))


def t2r(T):
    """
    Return rotational submatrix of a homogeneous transformation.
    @type T: 4x4 homogeneous transformation
    @param T: the transform matrix to convert
    @rtype: 3x3 orthonormal rotation matrix
    @return: rotation submatrix
    """    
    
    if ishomog(T)==False:
        error( 'input must be a homogeneous transform')
    return T[0:3,0:3]


def r2t(R):
    """
    Convert a 3x3 orthonormal rotation matrix to a 4x4 homogeneous transformation::
    
        T = | R 0 |
            | 0 1 |
            
    @type R: 3x3 orthonormal rotation matrix
    @param R: the rotation matrix to convert
    @rtype: 4x4 homogeneous matrix
    @return: homogeneous equivalent
    """
    
    return concatenate( (concatenate( (R, zeros((3,1))),1), mat([0,0,0,1])) )
