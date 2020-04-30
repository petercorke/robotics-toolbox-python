"""
Quaternion class.

Python implementation by: Luis Fernando Lara Tobar and Peter Corke.
Based on original Robotics Toolbox for Matlab code by Peter Corke.
Permission to use and copy is granted provided that acknowledgement of
the authors is made.

@author: Luis Fernando Lara Tobar and Peter Corke
"""

from numpy import *
from utility import *
import transform as T
import copy

# Copyright (C) 1999-2002, by Peter I. Corke

class quaternion:
    """A quaternion is a compact method of representing a 3D rotation that has
    computational advantages including speed and numerical robustness.

    A quaternion has 2 parts, a scalar s, and a vector v and is typically written::

        q = s <vx vy vz>

    A unit quaternion is one for which M{s^2+vx^2+vy^2+vz^2 = 1}.

    A quaternion can be considered as a rotation about a vector in space where
    q = cos (theta/2) sin(theta/2) <vx vy vz>
    where <vx vy vz> is a unit vector.

    Various functions such as INV, NORM, UNIT and PLOT are overloaded for
    quaternion objects.

    Arithmetic operators are also overloaded to allow quaternion multiplication,
    division, exponentiaton, and quaternion-vector multiplication (rotation).
    """

    def __init__(self, *args):
        '''
Constructor for quaternion objects:

    - q = quaternion(v, theta)    from vector plus angle
    - q = quaternion(R)       from a 3x3 or 4x4 matrix
    - q = quaternion(q)       from another quaternion
    - q = quaternion(s)       from a scalar
    - q = quaternion(v)       from a matrix/array/list v = [s v1 v2 v3]
    - q = quaternion(s, v1, v2, v3)    from 4 elements
    - q = quaternion(s, v)    from 4 elements
'''

        self.vec = [];
        
        if len(args) == 0:
                # default is a null rotation
                self.s = 1.0
                self.v = matrix([0.0, 0.0, 0.0])

        elif len(args) == 1:
            arg = args[0]
            
            if isinstance(arg, quaternion):
            # Q = QUATERNION(q) from another quaternion
                self.s = arg.s
                self.v = arg.v
                return
                
            if type(arg) is matrix:
                # Q = QUATERNION(R) from a 3x3
                if (arg.shape == (3,3)):
                    self.tr2q(arg);
                    return;

                # Q = QUATERNION(R) from a 4x4
                if (arg.shape == (4,4)):
                    self.tr2q(arg[0:3,0:3]);
                    return;
                    
            # some kind of list, vector, scalar...
            
            v = arg2array(arg);
            
            if len(v) == 4:
                # Q = QUATERNION([s v1 v2 v3]) from 4 elements
                self.s = v[0];
                self.v = mat(v[1:4]);
                
            elif len(v) == 3:
                self.s = 0
                self.v = mat(v[0:3]);

            elif len(v) == 1:
                # Q = QUATERNION(s) from a scalar
                #Q = QUATERNION(s)               from a scalar
                self.s = v[0];
                self.v = mat([0, 0, 0])

        elif len(args) == 2:
            # Q = QUATERNION(v, theta) from vector plus angle
            # Q = quaternion(s, v);

            a1 = arg2array(args[0]);
            a2 = arg2array(args[1]);
            
            if len(a1) == 1 and len(a2) == 3:
                # s, v
                self.s = a1[0];
                self.v = mat(a2);
            elif len(a1) == 3 and len(a2) == 1:
                # v, theta
                self.s = a2[0];
                self.v = mat(a1);

        elif len(args) == 4:
            self.s = args[0];
            self.v = mat(args[1:4])

        else:
                print "error"
                return None

    def __repr__(self):
            return "%f <%f, %f, %f>" % (self.s, self.v[0,0], self.v[0,1], self.v[0,2])


    def tr2q(self, t):
        #TR2Q   Convert homogeneous transform to a unit-quaternion
        #
        #   Q = tr2q(T)
        #
        #   Return a unit quaternion corresponding to the rotational part of the
        #   homogeneous transform T.

        qs = sqrt(trace(t)+1)/2.0
        kx = t[2,1] - t[1,2]    # Oz - Ay
        ky = t[0,2] - t[2,0]    # Ax - Nz
        kz = t[1,0] - t[0,1]    # Ny - Ox

        if (t[0,0] >= t[1,1]) and (t[0,0] >= t[2,2]):
                kx1 = t[0,0] - t[1,1] - t[2,2] + 1      # Nx - Oy - Az + 1
                ky1 = t[1,0] + t[0,1]           # Ny + Ox
                kz1 = t[2,0] + t[0,2]           # Nz + Ax
                add = (kx >= 0)
        elif (t[1,1] >= t[2,2]):
                kx1 = t[1,0] + t[0,1]           # Ny + Ox
                ky1 = t[1,1] - t[0,0] - t[2,2] + 1  # Oy - Nx - Az + 1
                kz1 = t[2,1] + t[1,2]           # Oz + Ay
                add = (ky >= 0)
        else:
                kx1 = t[2,0] + t[0,2]           # Nz + Ax
                ky1 = t[2,1] + t[1,2]           # Oz + Ay
                kz1 = t[2,2] - t[0,0] - t[1,1] + 1  # Az - Nx - Oy + 1
                add = (kz >= 0)

        if add:
                kx = kx + kx1
                ky = ky + ky1
                kz = kz + kz1
        else:
                kx = kx - kx1
                ky = ky - ky1
                kz = kz - kz1

        kv = matrix([kx, ky, kz])
        nm = linalg.norm( kv )
        if nm == 0:
                self.s = 1.0
                self.v = matrix([0.0, 0.0, 0.0])

        else:
                self.s = qs
                self.v = (sqrt(1 - qs**2) / nm) * kv

    ############### OPERATORS #########################################
    #PLUS Add two quaternion objects
    #
    # Invoked by the + operator
    #
    # q1+q2 standard quaternion addition
    def __add__(self, q):
        '''
        Return a new quaternion that is the element-wise sum of the operands.
        '''
        if isinstance(q, quaternion):
            qr = quaternion()
            qr.s = 0

            qr.s = self.s + q.s
            qr.v = self.v + q.v

            return qr
        else:
            raise ValueError

    #MINUS Subtract two quaternion objects
    #
    # Invoked by the - operator
    #
    # q1-q2 standard quaternion subtraction

    def __sub__(self, q):
        '''
        Return a new quaternion that is the element-wise difference of the operands.
        '''
        if isinstance(q, quaternion):
            qr = quaternion()
            qr.s = 0

            qr.s = self.s - q.s
            qr.v = self.v - q.v

            return qr
        else:
            raise ValueError

    # q * q  or q * const
    def __mul__(self, q2):
        '''
        Quaternion product. Several cases are handled
        
            - q * q   quaternion multiplication
            - q * c   element-wise multiplication by constant
            - q * v   quaternion-vector multiplication q * v * q.inv();
        '''
        qr = quaternion();
        
        if isinstance(q2, quaternion):
                
            #Multiply unit-quaternion by unit-quaternion
            #
            #   QQ = qqmul(Q1, Q2)

            # decompose into scalar and vector components
            s1 = self.s;    v1 = self.v
            s2 = q2.s;  v2 = q2.v

            # form the product
            qr.s = s1*s2 - v1*v2.T
            qr.v = s1*v2 + s2*v1 + cross(v1,v2)

        elif type(q2) is matrix:
                
            # Multiply vector by unit-quaternion
            #
            #   Rotate the vector V by the unit-quaternion Q.

            if q2.shape == (1,3) or q2.shape == (3,1):
                    qr = self * quaternion(q2) * self.inv()
                    return qr.v;
            else:
                    raise ValueError;

        else:
            qr.s = self.s * q2
            qr.v = self.v * q2

        return qr

    def __rmul__(self, c):
        '''
        Quaternion product. Several cases are handled
 
            - c * q   element-wise multiplication by constant
        '''
        qr = quaternion()
        qr.s = self.s * c
        qr.v = self.v * c

        return qr
        
    def __imul__(self, x):
        '''
        Quaternion in-place multiplication
        
            - q *= q2
            
        '''
        
        if isinstance(x, quaternion):
            s1 = self.s;   
            v1 = self.v
            s2 = x.s
            v2 = x.v

            # form the product
            self.s = s1*s2 - v1*v2.T
            self.v = s1*v2 + s2*v1 + cross(v1,v2)

        elif isscalar(x):
            self.s *= x;
            self.v *= x;

        return self;


    def __div__(self, q):
        '''Return quaternion quotient.  Several cases handled:
            - q1 / q2      quaternion division implemented as q1 * q2.inv()
            - q1 / c       element-wise division
        '''
        if isinstance(q, quaternion):
            qr = quaternion()
            qr = self * q.inv()
        elif isscalar(q):
            qr.s = self.s / q
            qr.v = self.v / q

        return qr


    def __pow__(self, p):
        '''
        Quaternion exponentiation.  Only integer exponents are handled.  Negative
        integer exponents are supported.
        '''
        
        # check that exponent is an integer
        if not isinstance(p, int):
            raise ValueError
        
        qr = quaternion()
        q = quaternion(self);
        
        # multiply by itself so many times
        for i in range(0, abs(p)):
            qr *= q

        # if exponent was negative, invert it
        if p < 0:
            qr = qr.inv()

        return qr

    def copy(self):
        """
        Return a copy of the quaternion.
        """
        return copy.copy(self);
                
    def inv(self):
        """Return the inverse.
        
        @rtype: quaternion
        @return: the inverse
        """
        
        qi = quaternion(self);
        qi.v = -qi.v;
        
        return qi;



    def norm(self):
        """Return the norm of this quaternion.
        
        @rtype: number
        @return: the norm
        """
        
        return linalg.norm(self.double())

    def double(self):
        """Return the quaternion as 4-element vector.
        
        @rtype: 4-vector
        @return: the quaternion elements
        """
        return concatenate( (mat(self.s), self.v), 1 )


    def unit(self):
        """Return an equivalent unit quaternion
        
        @rtype: quaternion
        @return: equivalent unit quaternion
        """
        
        qr = quaternion()
        nm = self.norm()

        qr.s = self.s / nm
        qr.v = self.v / nm

        return qr


    def tr(self):
        """Return an equivalent rotation matrix.
        
        @rtype: 4x4 homogeneous transform
        @return: equivalent rotation matrix
        """

        return T.r2t( self.r() )

    def r(self):
        """Return an equivalent rotation matrix.
        
        @rtype: 3x3 orthonormal rotation matrix
        @return: equivalent rotation matrix
        """

        s = self.s;
        x = self.v[0,0]
        y = self.v[0,1]
        z = self.v[0,2]

        return matrix([[ 1-2*(y**2+z**2),   2*(x*y-s*z),    2*(x*z+s*y)],
                       [2*(x*y+s*z),    1-2*(x**2+z**2),    2*(y*z-s*x)],
                       [2*(x*z-s*y),    2*(y*z+s*x),    1-2*(x**2+y**2)]])

    

#QINTERP Interpolate rotations expressed by quaternion objects
#
#   QI = qinterp(Q1, Q2, R)
#
# Return a unit-quaternion that interpolates between Q1 and Q2 as R moves
# from 0 to 1.  This is a spherical linear interpolation (slerp) that can
# be interpretted as interpolation along a great circle arc on a sphere.
#
# If r is a vector, QI, is a cell array of quaternions, each element
# corresponding to sequential elements of R.
#
# See also: CTRAJ, QUATERNION.

# MOD HISTORY
# 2/99 convert to use of objects
# $Log: qinterp.m,v $
# Revision 1.3  2002/04/14 11:02:54  pic
# Changed see also line.
#
# Revision 1.2  2002/04/01 12:06:48  pic
# General tidyup, help comments, copyright, see also, RCS keys.
#
# $Revision: 1.3 $
#
# Copyright (C) 1999-2002, by Peter I. Corke

    def interp(Q1, Q2, r):
        q1 = Q1.double()
        q2 = Q2.double()

        theta = arccos(q1*q2.T)
        q = []
        count = 0

        if isscalar(r):
            if r<0 or r>1:
                raise 'R out of range'
            if theta == 0:
                q = quaternion(Q1)
            else:
                q = quaternion( (sin((1-r)*theta) * q1 + sin(r*theta) * q2) / sin(theta) )
        else:
            for R in r:
                if theta == 0:
                    qq = Q1
                else:
                    qq = quaternion( (sin((1-R)*theta) * q1 + sin(R*theta) * q2) / sin(theta) )
                q.append(qq)
        return q

